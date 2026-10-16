use anyhow::Result;
use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

use crate::flags::{merge, ModifierState};
use crate::keymap::scancode_to_keycode;

pub struct KeyboardInjector {
    state: ModifierState,
}

/// Set the flags of `event` to `ours`, keeping the bits that the injector does not manage.
pub(crate) fn set_event_flags(event: &CGEvent, ours: u64) {
    let flags = merge(event.get_flags().bits(), ours);
    event.set_flags(CGEventFlags::from_bits_retain(flags));
}

impl KeyboardInjector {
    pub fn new() -> Result<Self> {
        // Verify we can create an event source (permission check)
        let _ = CGEventSource::new(CGEventSourceStateID::HIDSystemState).map_err(|_| {
            anyhow::anyhow!("Failed to create CGEventSource — check Accessibility permission")
        })?;
        Ok(Self {
            state: ModifierState::new(),
        })
    }

    fn source() -> Result<CGEventSource> {
        CGEventSource::new(CGEventSourceStateID::HIDSystemState)
            .map_err(|_| anyhow::anyhow!("Failed to create CGEventSource"))
    }

    /// Flags of the client's held modifiers (for mouse events).
    pub fn modifier_flags(&self) -> u64 {
        self.state.flags()
    }

    /// Inject a key press or release event
    pub fn inject_key(&mut self, scancode: u8, extended: bool, pressed: bool) -> Result<()> {
        let keycode = match scancode_to_keycode(scancode, extended) {
            Some(kc) => kc,
            None => {
                tracing::warn!(scancode, extended, "Unknown scancode, ignoring");
                return Ok(());
            }
        };

        self.state.key(keycode, pressed);
        let source = Self::source()?;
        let event = CGEvent::new_keyboard_event(source, keycode as CGKeyCode, pressed)
            .map_err(|_| anyhow::anyhow!("Failed to create keyboard event"))?;
        // The created event has the flags of the HID system state; set the client's instead.
        set_event_flags(&event, self.state.key_flags(keycode));

        event.post(CGEventTapLocation::HID);
        tracing::trace!(
            scancode,
            keycode,
            pressed,
            flags = format_args!("{:#x}", event.get_flags().bits()),
            "Keyboard event injected"
        );
        Ok(())
    }

    /// Inject a unicode character press/release
    pub fn inject_unicode(&mut self, ch: u16, pressed: bool) -> Result<()> {
        let source = Self::source()?;
        let event = CGEvent::new_keyboard_event(source, 0, pressed)
            .map_err(|_| anyhow::anyhow!("Failed to create unicode event"))?;
        set_event_flags(&event, self.state.flags());

        if pressed {
            event.set_string_from_utf16_unchecked(&[ch]);
        }

        event.post(CGEventTapLocation::HID);
        Ok(())
    }
}
