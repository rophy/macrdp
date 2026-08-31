use ironrdp_server::{KeyboardEvent, MouseEvent, RdpServerInputHandler};
use macrdp_input::{KeyboardInjector, MouseButton, MouseInjector};
use std::sync::{Arc, Mutex};

/// Maps RDP desktop coordinates to macOS logical coordinates.
///
/// Completely independent of capture/encode resolution — only cares about
/// the RDP desktop size (what the client thinks) and the macOS logical
/// display size (what CGEvent expects).
///
/// Formula: `mac = rdp × logical ÷ rdp_desktop`
#[derive(Debug, Clone)]
pub struct MouseCoordMapper(Arc<Mutex<CoordState>>);

#[derive(Debug)]
struct CoordState {
    logical_w: f64,
    logical_h: f64,
    rdp_w: f64,
    rdp_h: f64,
}

impl MouseCoordMapper {
    pub fn new(logical_w: u16, logical_h: u16, rdp_w: u16, rdp_h: u16) -> Self {
        tracing::info!(
            logical_w, logical_h, rdp_w, rdp_h,
            "MouseCoordMapper initialized"
        );
        Self(Arc::new(Mutex::new(CoordState {
            logical_w: logical_w as f64,
            logical_h: logical_h as f64,
            rdp_w: rdp_w.max(1) as f64,
            rdp_h: rdp_h.max(1) as f64,
        })))
    }

    pub fn map(&self, rdp_x: u16, rdp_y: u16) -> (u16, u16) {
        let s = self.0.lock().unwrap();
        let mac_x = (rdp_x as f64 * s.logical_w / s.rdp_w) as u16;
        let mac_y = (rdp_y as f64 * s.logical_h / s.rdp_h) as u16;
        (mac_x, mac_y)
    }

    pub fn update_rdp_size(&self, rdp_w: u16, rdp_h: u16) {
        let mut s = self.0.lock().unwrap();
        let old_w = s.rdp_w;
        let old_h = s.rdp_h;
        s.rdp_w = rdp_w.max(1) as f64;
        s.rdp_h = rdp_h.max(1) as f64;
        tracing::info!(
            old_rdp_w = old_w, old_rdp_h = old_h,
            new_rdp_w = rdp_w, new_rdp_h = rdp_h,
            logical_w = s.logical_w, logical_h = s.logical_h,
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
    last_rdp_x: u16,
    last_rdp_y: u16,
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
            tracing::warn!(
                "Input injection may fail — ensure Accessibility permission is granted"
            );
        }

        Self {
            keyboard,
            mouse,
            last_mouse_x: 0,
            last_mouse_y: 0,
            last_rdp_x: 0,
            last_rdp_y: 0,
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
                self.last_rdp_x = x;
                self.last_rdp_y = y;
                m.move_to(mx, my)
            }
            MouseEvent::LeftPressed => {
                tracing::debug!(
                    rdp_x = self.last_rdp_x, rdp_y = self.last_rdp_y,
                    mac_x = self.last_mouse_x, mac_y = self.last_mouse_y,
                    "Mouse click (left down)"
                );
                m.button_event(MouseButton::Left, true, self.last_mouse_x, self.last_mouse_y)
            }
            MouseEvent::LeftReleased => {
                m.button_event(MouseButton::Left, false, self.last_mouse_x, self.last_mouse_y)
            }
            MouseEvent::RightPressed => {
                m.button_event(MouseButton::Right, true, self.last_mouse_x, self.last_mouse_y)
            }
            MouseEvent::RightReleased => {
                m.button_event(MouseButton::Right, false, self.last_mouse_x, self.last_mouse_y)
            }
            MouseEvent::MiddlePressed => {
                m.button_event(MouseButton::Middle, true, self.last_mouse_x, self.last_mouse_y)
            }
            MouseEvent::MiddleReleased => {
                m.button_event(MouseButton::Middle, false, self.last_mouse_x, self.last_mouse_y)
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
