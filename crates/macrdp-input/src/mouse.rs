use anyhow::Result;
use core_graphics::event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton, EventField};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use core_graphics::geometry::CGPoint;
use foreign_types::ForeignType;
use std::time::{Duration, Instant};

use crate::keyboard::set_event_flags;

extern "C" {
    fn CGEventCreateScrollWheelEvent2(
        source: *mut core_graphics::sys::CGEventSource,
        units: u32,
        wheel_count: u32,
        wheel1: i32,
        wheel2: i32,
        wheel3: i32,
    ) -> *mut core_graphics::sys::CGEvent;
}

/// ScrollEventUnit::Line
const SCROLL_UNIT_LINE: u32 = 1;

/// RDP wheel rotation units for one notch (WHEEL_DELTA).
const WHEEL_DELTA: i32 = 120;

/// Lines scrolled per notch: the Windows default (SPI_GETWHEELSCROLLLINES = 3).
const LINES_PER_NOTCH: i32 = 3;

/// Turns RDP wheel rotation units (120 per notch, less for high-resolution
/// wheels and touchpads) into whole lines. The rest is kept for the next
/// event in the same direction, so small deltas add up like on Windows.
#[derive(Debug, Default)]
pub struct WheelAccumulator {
    /// Rest in units of 1/WHEEL_DELTA line.
    rest: i32,
}

impl WheelAccumulator {
    /// `value`: rotation units, positive = up/right. Returns the lines to scroll (same sign).
    pub fn lines(&mut self, value: i16) -> i32 {
        let units = value as i32 * LINES_PER_NOTCH;
        if units.signum() * self.rest.signum() < 0 {
            self.rest = 0;
        }
        self.rest += units;
        let lines = self.rest / WHEEL_DELTA;
        self.rest -= lines * WHEEL_DELTA;
        lines
    }
}

/// Largest move (points, each axis) between the clicks of a double click.
/// Windows uses a 4x4 rectangle (SM_CXDOUBLECLK).
const CLICK_DISTANCE: f64 = 4.0;

/// Counts multiple clicks (double, triple) for `kCGMouseEventClickState`.
/// RDP sends only presses and releases; macOS apps read the click count
/// from the event, so the injector must set it.
#[derive(Debug, Default)]
pub struct ClickCounter {
    last: Option<LastClick>,
}

#[derive(Debug, Clone, Copy)]
struct LastClick {
    button: MouseButton,
    at: Instant,
    x: f64,
    y: f64,
    count: i64,
}

impl ClickCounter {
    /// A press of `button` at (x, y): returns its click count (1 = single click).
    pub fn press(
        &mut self,
        button: MouseButton,
        x: f64,
        y: f64,
        now: Instant,
        interval: Duration,
    ) -> i64 {
        let count = match self.last {
            Some(l)
                if l.button == button
                    && now.duration_since(l.at) <= interval
                    && (x - l.x).abs() <= CLICK_DISTANCE
                    && (y - l.y).abs() <= CLICK_DISTANCE =>
            {
                l.count + 1
            }
            _ => 1,
        };
        self.last = Some(LastClick {
            button,
            at: now,
            x,
            y,
            count,
        });
        count
    }

    /// The click count for a release of `button` (same as its press).
    pub fn release(&self, button: MouseButton) -> i64 {
        match self.last {
            Some(l) if l.button == button => l.count,
            _ => 1,
        }
    }
}

/// The user's double-click time (System Settings).
fn double_click_interval() -> Duration {
    let secs = objc2_app_kit::NSEvent::doubleClickInterval();
    if secs.is_finite() && secs > 0.0 {
        Duration::from_secs_f64(secs)
    } else {
        Duration::from_millis(500)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollAxis {
    Vertical,
    Horizontal,
}

pub struct MouseInjector {
    vertical: WheelAccumulator,
    horizontal: WheelAccumulator,
    clicks: ClickCounter,
}

impl MouseInjector {
    pub fn new() -> Result<Self> {
        let _ = CGEventSource::new(CGEventSourceStateID::HIDSystemState).map_err(|_| {
            anyhow::anyhow!("Failed to create CGEventSource — check Accessibility permission")
        })?;
        Ok(Self {
            vertical: WheelAccumulator::default(),
            horizontal: WheelAccumulator::default(),
            clicks: ClickCounter::default(),
        })
    }

    fn source() -> Result<CGEventSource> {
        CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| anyhow::anyhow!("Failed to create CGEventSource"))
    }

    /// `flags`: the client's held modifiers (`KeyboardInjector::modifier_flags`).
    pub fn move_to(&self, x: u16, y: u16, flags: u64) -> Result<()> {
        let point = CGPoint::new(x as f64, y as f64);
        let source = Self::source()?;
        let event =
            CGEvent::new_mouse_event(source, CGEventType::MouseMoved, point, CGMouseButton::Left)
                .map_err(|_| anyhow::anyhow!("Failed to create mouse move event"))?;
        set_event_flags(&event, flags);

        event.post(CGEventTapLocation::HID);
        tracing::trace!(x, y, "Mouse moved");
        Ok(())
    }

    pub fn button_event(
        &mut self,
        button: MouseButton,
        pressed: bool,
        x: u16,
        y: u16,
        flags: u64,
    ) -> Result<()> {
        let point = CGPoint::new(x as f64, y as f64);
        let (event_type, cg_button) = match (button, pressed) {
            (MouseButton::Left, true) => (CGEventType::LeftMouseDown, CGMouseButton::Left),
            (MouseButton::Left, false) => (CGEventType::LeftMouseUp, CGMouseButton::Left),
            (MouseButton::Right, true) => (CGEventType::RightMouseDown, CGMouseButton::Right),
            (MouseButton::Right, false) => (CGEventType::RightMouseUp, CGMouseButton::Right),
            (MouseButton::Middle, true) => (CGEventType::OtherMouseDown, CGMouseButton::Center),
            (MouseButton::Middle, false) => (CGEventType::OtherMouseUp, CGMouseButton::Center),
        };

        let source = Self::source()?;
        let event = CGEvent::new_mouse_event(source, event_type, point, cg_button)
            .map_err(|_| anyhow::anyhow!("Failed to create mouse button event"))?;
        set_event_flags(&event, flags);
        let clicks = if pressed {
            self.clicks.press(
                button,
                point.x,
                point.y,
                Instant::now(),
                double_click_interval(),
            )
        } else {
            self.clicks.release(button)
        };
        event.set_integer_value_field(EventField::MOUSE_EVENT_CLICK_STATE, clicks);

        event.post(CGEventTapLocation::HID);
        tracing::trace!(?button, pressed, x, y, clicks, "Mouse button event");
        Ok(())
    }

    /// Scroll by an RDP wheel event. `value`: rotation units (120 per notch),
    /// positive = wheel forward (up) or tilt right.
    pub fn scroll(&mut self, axis: ScrollAxis, value: i16, flags: u64) -> Result<()> {
        let lines = match axis {
            ScrollAxis::Vertical => self.vertical.lines(value),
            ScrollAxis::Horizontal => self.horizontal.lines(value),
        };
        if lines == 0 {
            tracing::trace!(?axis, value, "Mouse scroll below one line, kept");
            return Ok(());
        }
        // CGEvent: wheel1 > 0 scrolls up, wheel2 > 0 scrolls left.
        let (wheel_count, wheel1, wheel2) = match axis {
            ScrollAxis::Vertical => (1, lines, 0),
            ScrollAxis::Horizontal => (2, 0, -lines),
        };
        unsafe {
            let event_ref = CGEventCreateScrollWheelEvent2(
                std::ptr::null_mut(),
                SCROLL_UNIT_LINE,
                wheel_count,
                wheel1,
                wheel2,
                0,
            );
            if event_ref.is_null() {
                return Err(anyhow::anyhow!("Failed to create scroll event"));
            }
            let event = CGEvent::from_ptr(event_ref);
            set_event_flags(&event, flags);
            event.post(CGEventTapLocation::HID);
        }
        tracing::trace!(?axis, value, lines, "Mouse scroll");
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[cfg(test)]
mod tests {
    use super::*;

    const DC: Duration = Duration::from_millis(500);

    #[test]
    fn quick_clicks_count_up() {
        let mut c = ClickCounter::default();
        let t = Instant::now();
        assert_eq!(c.press(MouseButton::Left, 100.0, 100.0, t, DC), 1);
        assert_eq!(c.release(MouseButton::Left), 1);
        let t2 = t + Duration::from_millis(110);
        assert_eq!(c.press(MouseButton::Left, 101.0, 99.0, t2, DC), 2);
        assert_eq!(c.release(MouseButton::Left), 2);
        let t3 = t2 + Duration::from_millis(110);
        assert_eq!(c.press(MouseButton::Left, 101.0, 99.0, t3, DC), 3);
    }

    #[test]
    fn slow_far_or_other_button_clicks_start_again() {
        let mut c = ClickCounter::default();
        let t = Instant::now();
        c.press(MouseButton::Left, 100.0, 100.0, t, DC);
        let late = t + Duration::from_millis(600);
        assert_eq!(c.press(MouseButton::Left, 100.0, 100.0, late, DC), 1);
        let t2 = late + Duration::from_millis(100);
        assert_eq!(c.press(MouseButton::Left, 110.0, 100.0, t2, DC), 1);
        let t3 = t2 + Duration::from_millis(100);
        assert_eq!(c.press(MouseButton::Right, 110.0, 100.0, t3, DC), 1);
        assert_eq!(c.release(MouseButton::Left), 1);
        let t4 = t3 + Duration::from_millis(100);
        assert_eq!(c.press(MouseButton::Right, 110.0, 100.0, t4, DC), 2);
    }

    #[test]
    fn one_notch_is_three_lines() {
        let mut w = WheelAccumulator::default();
        assert_eq!(w.lines(120), 3);
        assert_eq!(w.lines(-120), -3);
        assert_eq!(w.lines(240), 6);
    }

    #[test]
    fn partial_deltas_add_up() {
        let mut w = WheelAccumulator::default();
        // 40 units = 1 line
        assert_eq!(w.lines(40), 1);
        // 15 units = 3/8 line each: lines come at the 3rd and 6th event
        let mut w = WheelAccumulator::default();
        let got: Vec<i32> = (0..8).map(|_| w.lines(15)).collect();
        assert_eq!(got, vec![0, 0, 1, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn rest_is_dropped_when_the_direction_changes() {
        let mut w = WheelAccumulator::default();
        assert_eq!(w.lines(30), 0); // rest 3/4 line up
        assert_eq!(w.lines(-20), 0); // rest dropped, 1/2 line down
        assert_eq!(w.lines(-20), -1);
    }
}
