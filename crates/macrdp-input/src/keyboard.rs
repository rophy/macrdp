use anyhow::Result;
use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation, CGKeyCode};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

use crate::flags::{merge, ModifierState};
use crate::keymap::{keypad_navigation_keycode, scancode_to_keycode, SCANCODE_NUM_LOCK};

pub struct KeyboardInjector {
    state: ModifierState,
    unicode: UnicodeJoiner,
}

/// What to post for an RDP Unicode keyboard event.
#[derive(Debug, PartialEq, Eq)]
pub enum UnicodeAction {
    /// Key down carrying these UTF-16 units (one character).
    Down(Vec<u16>),
    /// Key up.
    Up,
    /// Nothing to post.
    None,
}

/// RDP sends one UTF-16 unit per Unicode event, so a character outside the
/// BMP (emoji) comes as a high and a low surrogate. A lone surrogate is not
/// valid text; join the pair into one key down.
#[derive(Debug, Default)]
pub struct UnicodeJoiner {
    high: Option<u16>,
}

impl UnicodeJoiner {
    pub fn feed(&mut self, unit: u16, pressed: bool) -> UnicodeAction {
        match unit {
            0xD800..=0xDBFF => {
                // High surrogate: wait for the low one; its release posts nothing.
                if pressed {
                    self.high = Some(unit);
                }
                UnicodeAction::None
            }
            0xDC00..=0xDFFF => match (pressed, self.high) {
                (true, Some(high)) => {
                    self.high = None;
                    UnicodeAction::Down(vec![high, unit])
                }
                (true, None) => {
                    tracing::debug!(unit, "Low surrogate without a high surrogate, dropped");
                    UnicodeAction::None
                }
                (false, _) => UnicodeAction::Up,
            },
            _ => {
                if pressed {
                    self.high = None;
                    UnicodeAction::Down(vec![unit])
                } else {
                    UnicodeAction::Up
                }
            }
        }
    }
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
            unicode: UnicodeJoiner::default(),
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

    /// Set Caps Lock and Num Lock from an RDP Synchronize event (sent on connect and focus).
    pub fn synchronize(&mut self, caps_lock: bool, num_lock: bool) {
        self.state.set_locks(caps_lock, num_lock);
        tracing::debug!(caps_lock, num_lock, "Keyboard locks synchronized");
    }

    /// Inject a key press or release event
    pub fn inject_key(&mut self, scancode: u8, extended: bool, pressed: bool) -> Result<()> {
        if scancode == SCANCODE_NUM_LOCK && !extended && pressed {
            self.state.toggle_num_lock();
        }
        // With Num Lock off the keypad keys move the cursor, as on Windows.
        let keypad_nav = if !extended && !self.state.num_lock() {
            keypad_navigation_keycode(scancode)
        } else {
            None
        };
        let keycode = match keypad_nav.or_else(|| scancode_to_keycode(scancode, extended)) {
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

    /// Inject a Unicode keyboard event (one UTF-16 unit) press/release
    pub fn inject_unicode(&mut self, unit: u16, pressed: bool) -> Result<()> {
        let (down, text) = match self.unicode.feed(unit, pressed) {
            UnicodeAction::Down(text) => (true, text),
            UnicodeAction::Up => (false, Vec::new()),
            UnicodeAction::None => return Ok(()),
        };
        let source = Self::source()?;
        let event = CGEvent::new_keyboard_event(source, 0, down)
            .map_err(|_| anyhow::anyhow!("Failed to create unicode event"))?;
        set_event_flags(&event, self.state.flags());

        if down {
            event.set_string_from_utf16_unchecked(&text);
        }

        event.post(CGEventTapLocation::HID);
        tracing::trace!(?text, down, "Unicode event injected");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmp_character_is_posted_as_is() {
        let mut j = UnicodeJoiner::default();
        assert_eq!(j.feed(0x00E9, true), UnicodeAction::Down(vec![0x00E9]));
        assert_eq!(j.feed(0x00E9, false), UnicodeAction::Up);
    }

    #[test]
    fn surrogate_pair_becomes_one_character() {
        // U+1F600 = D83D DE00; order: high down, high up, low down, low up
        let mut j = UnicodeJoiner::default();
        assert_eq!(j.feed(0xD83D, true), UnicodeAction::None);
        assert_eq!(j.feed(0xD83D, false), UnicodeAction::None);
        assert_eq!(
            j.feed(0xDE00, true),
            UnicodeAction::Down(vec![0xD83D, 0xDE00])
        );
        assert_eq!(j.feed(0xDE00, false), UnicodeAction::Up);
    }

    #[test]
    fn surrogate_pair_with_nested_releases() {
        let mut j = UnicodeJoiner::default();
        assert_eq!(j.feed(0xD83D, true), UnicodeAction::None);
        assert_eq!(
            j.feed(0xDE00, true),
            UnicodeAction::Down(vec![0xD83D, 0xDE00])
        );
        assert_eq!(j.feed(0xDE00, false), UnicodeAction::Up);
        assert_eq!(j.feed(0xD83D, false), UnicodeAction::None);
    }

    #[test]
    fn lone_low_surrogate_is_dropped() {
        let mut j = UnicodeJoiner::default();
        assert_eq!(j.feed(0xDE00, true), UnicodeAction::None);
        // a high surrogate followed by a BMP character is dropped too
        assert_eq!(j.feed(0xD83D, true), UnicodeAction::None);
        assert_eq!(j.feed(0x0041, true), UnicodeAction::Down(vec![0x0041]));
        assert_eq!(j.feed(0xDE00, true), UnicodeAction::None);
    }
}
