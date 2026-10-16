use ironrdp_server::{
    KeyboardEvent, MouseButton as RdpMouseButton, MouseEvent, RdpServerInputHandler,
};
use macrdp_input::{KeyboardInjector, MouseButton, MouseInjector};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How many recent client-caused cursor positions are kept to recognise an echo.
/// The cursor tick (16 ms) reads the position after some of the client's moves
/// were posted, so it can see any of the last few, not only the newest one.
const RECENT_CLIENT_MOVES: usize = 32;

/// An older client move counts as the cause of the cursor position only this
/// long. Later, the same position comes from a local move and must be sent.
const CLIENT_MOVE_WINDOW: Duration = Duration::from_millis(250);

/// Maps RDP desktop coordinates to macOS logical coordinates.
///
/// Only cares about the RDP desktop size (what the client thinks) and the
/// macOS logical display size (what CGEvent expects). The capture scales the
/// display into the desktop with its aspect ratio kept and centred
/// (ScreenCaptureKit `scalesToFit`), so a desktop of another aspect ratio has
/// black bands; the mapping uses the same picture rectangle.
///
/// Formula: `mac = (rdp − band) ÷ scale`, `scale = min(rdp_w ÷ logical_w, rdp_h ÷ logical_h)`
#[derive(Debug, Clone)]
pub struct MouseCoordMapper(Arc<Mutex<CoordState>>);

#[derive(Debug)]
struct CoordState {
    logical_w: f64,
    logical_h: f64,
    rdp_w: f64,
    rdp_h: f64,
    /// Mac positions the client's own input moved the cursor to (newest last)
    recent_client_moves: VecDeque<((u16, u16), Instant)>,
}

impl CoordState {
    /// Scale and offsets of the display picture in the RDP desktop:
    /// `rdp = mac × scale + offset`.
    fn picture(&self) -> (f64, f64, f64) {
        let scale = (self.rdp_w / self.logical_w).min(self.rdp_h / self.logical_h);
        let off_x = (self.rdp_w - self.logical_w * scale) / 2.0;
        let off_y = (self.rdp_h - self.logical_h * scale) / 2.0;
        (scale, off_x, off_y)
    }
}

impl MouseCoordMapper {
    pub fn new(logical_w: u16, logical_h: u16, rdp_w: u16, rdp_h: u16) -> Self {
        tracing::info!(
            logical_w,
            logical_h,
            rdp_w,
            rdp_h,
            "MouseCoordMapper initialized"
        );
        Self(Arc::new(Mutex::new(CoordState {
            logical_w: logical_w as f64,
            logical_h: logical_h as f64,
            rdp_w: rdp_w.max(1) as f64,
            rdp_h: rdp_h.max(1) as f64,
            recent_client_moves: VecDeque::with_capacity(RECENT_CLIENT_MOVES),
        })))
    }

    pub fn map(&self, rdp_x: u16, rdp_y: u16) -> (u16, u16) {
        let s = self.0.lock().unwrap();
        let (scale, off_x, off_y) = s.picture();
        // A point in a black band goes to the nearest display edge.
        let mac_x = ((rdp_x as f64 - off_x) / scale).clamp(0.0, s.logical_w - 1.0);
        let mac_y = ((rdp_y as f64 - off_y) / scale).clamp(0.0, s.logical_h - 1.0);
        (mac_x as u16, mac_y as u16)
    }

    /// Maps a macOS logical position to RDP desktop coordinates (the inverse of
    /// [`Self::map`]), for PointerPosition updates sent to the client.
    pub fn to_rdp(&self, mac_x: u16, mac_y: u16) -> (u16, u16) {
        let s = self.0.lock().unwrap();
        let (scale, off_x, off_y) = s.picture();
        let x = (mac_x as f64 * scale + off_x).round();
        let y = (mac_y as f64 * scale + off_y).round();
        (
            x.min(s.rdp_w - 1.0).max(0.0) as u16,
            y.min(s.rdp_h - 1.0).max(0.0) as u16,
        )
    }

    /// Remember that client input moved the Mac cursor to `mac`.
    pub fn note_client_move(&self, mac: (u16, u16)) {
        self.note_client_move_at(mac, Instant::now());
    }

    fn note_client_move_at(&self, mac: (u16, u16), now: Instant) {
        let mut s = self.0.lock().unwrap();
        if let Some(last) = s.recent_client_moves.back_mut() {
            if last.0 == mac {
                last.1 = now;
                return;
            }
        }
        if s.recent_client_moves.len() == RECENT_CLIENT_MOVES {
            s.recent_client_moves.pop_front();
        }
        s.recent_client_moves.push_back((mac, now));
    }

    /// Whether the Mac cursor at `mac` was put there by recent client input.
    ///
    /// The server must not send such a position back as a PointerPosition:
    /// the client already shows its cursor there, and when the client scales
    /// the picture (mstsc smart sizing) it moves its cursor to the rounded
    /// position and sends that as a new move. The cursor then creeps until the
    /// round trip is exact (up to ~47 px seen with mstsc).
    ///
    /// The newest client position always counts (the client shows its cursor
    /// there). An older one counts only within [`CLIENT_MOVE_WINDOW`], so a local
    /// move to a place the client visited before is still sent.
    pub fn is_client_move(&self, mac: (u16, u16)) -> bool {
        self.is_client_move_at(mac, Instant::now())
    }

    fn is_client_move_at(&self, mac: (u16, u16), now: Instant) -> bool {
        let s = self.0.lock().unwrap();
        let mut moves = s.recent_client_moves.iter().rev();
        if moves.next().is_some_and(|(pos, _)| *pos == mac) {
            return true;
        }
        moves.any(|(pos, at)| *pos == mac && now.duration_since(*at) <= CLIENT_MOVE_WINDOW)
    }

    pub fn update_rdp_size(&self, rdp_w: u16, rdp_h: u16) {
        let mut s = self.0.lock().unwrap();
        let old_w = s.rdp_w;
        let old_h = s.rdp_h;
        s.rdp_w = rdp_w.max(1) as f64;
        s.rdp_h = rdp_h.max(1) as f64;
        tracing::info!(
            old_rdp_w = old_w,
            old_rdp_h = old_h,
            new_rdp_w = rdp_w,
            new_rdp_h = rdp_h,
            logical_w = s.logical_w,
            logical_h = s.logical_h,
            "MouseCoordMapper: RDP desktop size updated"
        );
    }
}

/// Bridges RDP input events to macOS CGEvent injection
pub struct MacInputHandler {
    keyboard: Option<KeyboardInjector>,
    mouse: Option<MouseInjector>,
    last_mouse_x: u16,
    last_mouse_y: u16,
    coord_mapper: MouseCoordMapper,
}

impl MacInputHandler {
    pub fn new(coord_mapper: MouseCoordMapper) -> Self {
        let keyboard = KeyboardInjector::new()
            .map_err(|e| tracing::error!("Failed to create keyboard injector: {e}"))
            .ok();
        let mouse = MouseInjector::new()
            .map_err(|e| tracing::error!("Failed to create mouse injector: {e}"))
            .ok();

        if keyboard.is_none() || mouse.is_none() {
            tracing::warn!("Input injection may fail — ensure Accessibility permission is granted");
        }

        Self {
            keyboard,
            mouse,
            last_mouse_x: 0,
            last_mouse_y: 0,
            coord_mapper,
        }
    }
}

impl RdpServerInputHandler for MacInputHandler {
    fn keyboard(&mut self, event: KeyboardEvent) {
        let Some(kb) = &self.keyboard else { return };

        let result = match event {
            KeyboardEvent::Pressed { code, extended } => kb.inject_key(code, extended, true),
            KeyboardEvent::Released { code, extended } => kb.inject_key(code, extended, false),
            KeyboardEvent::UnicodePressed(ch) => kb.inject_unicode(ch, true),
            KeyboardEvent::UnicodeReleased(ch) => kb.inject_unicode(ch, false),
            KeyboardEvent::Synchronize(_flags) => {
                tracing::debug!("Keyboard synchronize event (ignored)");
                Ok(())
            }
        };

        if let Err(e) = result {
            tracing::warn!("Keyboard injection failed: {e}");
        }
    }

    fn mouse(&mut self, event: MouseEvent) {
        let Some(m) = &self.mouse else { return };

        let result = match event {
            MouseEvent::Move { x, y } => {
                let (mx, my) = self.coord_mapper.map(x, y);
                self.last_mouse_x = mx;
                self.last_mouse_y = my;
                self.coord_mapper.note_client_move((mx, my));
                m.move_to(mx, my)
            }
            MouseEvent::Button {
                x,
                y,
                button,
                pressed,
            } => {
                let button = match button {
                    RdpMouseButton::Left => MouseButton::Left,
                    RdpMouseButton::Right => MouseButton::Right,
                    RdpMouseButton::Middle => MouseButton::Middle,
                    other => {
                        tracing::debug!(button = ?other, pressed, "Mouse button not supported (ignored)");
                        return;
                    }
                };
                // The event carries its position; the button event is posted there.
                let (mx, my) = self.coord_mapper.map(x, y);
                self.last_mouse_x = mx;
                self.last_mouse_y = my;
                self.coord_mapper.note_client_move((mx, my));
                m.button_event(button, pressed, mx, my)
            }
            MouseEvent::VerticalScroll { value } => m.scroll(value),
            _ => {
                tracing::trace!(?event, "Unhandled mouse event");
                Ok(())
            }
        };

        if let Err(e) = result {
            tracing::warn!("Mouse injection failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_moves_are_recognised_as_echo() {
        let m = MouseCoordMapper::new(1280, 720, 1280, 720);
        m.note_client_move((625, 190));
        m.note_client_move((626, 191));
        assert!(m.is_client_move((625, 190)));
        assert!(m.is_client_move((626, 191)));
        // A position the client never caused (local user, app warp) is not an echo.
        assert!(!m.is_client_move((625, 188)));
    }

    #[test]
    fn an_old_client_position_is_not_an_echo() {
        let m = MouseCoordMapper::new(1280, 720, 1280, 720);
        let t0 = Instant::now();
        m.note_client_move_at((640, 360), t0);
        m.note_client_move_at((102, 458), t0 + Duration::from_millis(10));
        // The tick may still see the older position shortly after.
        assert!(m.is_client_move_at((640, 360), t0 + Duration::from_millis(100)));
        // Seconds later a local move back to (640, 360) must reach the client.
        assert!(!m.is_client_move_at((640, 360), t0 + Duration::from_secs(5)));
        // The newest client position never expires.
        assert!(m.is_client_move_at((102, 458), t0 + Duration::from_secs(5)));
    }

    #[test]
    fn to_rdp_maps_mac_position_into_the_rdp_desktop() {
        // 900x560 session on a 1280x720 Mac: a Mac position must not be sent unscaled.
        let m = MouseCoordMapper::new(1280, 720, 900, 560);
        assert_eq!(m.to_rdp(640, 360), (450, 280));
        // Picture rows 26.875..533.125 (black bands above and below).
        assert_eq!(m.to_rdp(0, 0), (0, 27));
        assert_eq!(m.to_rdp(1279, 719), (899, 532));
    }

    #[test]
    fn map_follows_the_letterboxed_picture() {
        // 16:9 Mac in a 16:10 session: mstsc showed black rows 0..25 and
        // 534..559, picture scale 900/1280 in both axes.
        let m = MouseCoordMapper::new(1280, 720, 900, 560);
        assert_eq!(m.map(450, 280), (640, 360));
        assert_eq!(m.map(450, 27), (640, 0));
        assert_eq!(m.map(450, 532), (640, 718));
        assert_eq!(m.map(72, 349), (102, 458));
        // In a band: clamped to the display edge.
        assert_eq!(m.map(0, 0), (0, 0));
        assert_eq!(m.map(899, 559), (1278, 719));
    }

    #[test]
    fn map_pillarboxes_a_wider_session() {
        // 16:9 Mac in a 2:1 session: bands left and right.
        let m = MouseCoordMapper::new(1280, 720, 1440, 720);
        assert_eq!(m.map(80, 0), (0, 0));
        assert_eq!(m.map(720, 360), (640, 360));
        assert_eq!(m.to_rdp(0, 0), (80, 0));
    }

    #[test]
    fn map_scales_a_session_of_the_same_aspect_ratio() {
        let m = MouseCoordMapper::new(1280, 720, 2560, 1440);
        assert_eq!(m.map(1001, 575), (500, 287));
        assert_eq!(m.to_rdp(500, 287), (1000, 574));
    }

    #[test]
    fn to_rdp_is_identity_at_native_size() {
        let m = MouseCoordMapper::new(1280, 720, 1280, 720);
        assert_eq!(m.to_rdp(500, 287), (500, 287));
    }

    #[test]
    fn only_recent_client_moves_are_kept() {
        let m = MouseCoordMapper::new(1280, 720, 1280, 720);
        for i in 0..=RECENT_CLIENT_MOVES as u16 {
            m.note_client_move((i, 0));
        }
        assert!(!m.is_client_move((0, 0)));
        assert!(m.is_client_move((1, 0)));
        assert!(m.is_client_move((RECENT_CLIENT_MOVES as u16, 0)));
    }
}
