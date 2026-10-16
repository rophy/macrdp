use anyhow::Result;
use core_graphics::event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use core_graphics::geometry::CGPoint;
use foreign_types::ForeignType;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollAxis {
    Vertical,
    Horizontal,
}

pub struct MouseInjector {
    vertical: WheelAccumulator,
    horizontal: WheelAccumulator,
}

impl MouseInjector {
    pub fn new() -> Result<Self> {
        let _ = CGEventSource::new(CGEventSourceStateID::HIDSystemState).map_err(|_| {
            anyhow::anyhow!("Failed to create CGEventSource — check Accessibility permission")
        })?;
        Ok(Self {
            vertical: WheelAccumulator::default(),
            horizontal: WheelAccumulator::default(),
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
        &self,
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

        event.post(CGEventTapLocation::HID);
        tracing::trace!(?button, pressed, x, y, "Mouse button event");
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

#[derive(Debug, Clone, Copy)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[cfg(test)]
mod tests {
    use super::*;

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
