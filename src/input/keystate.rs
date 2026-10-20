//! Pure keyboard state for the input handler: the client's Num Lock, the
//! keycode each held key went down with, and RDP Unicode key events.
//!
//! No platform calls, so it is unit-tested on every target. Ported from
//! macrdp-v1 (crates/macrdp-input/src/flags.rs and keyboard.rs, commits
//! d74b3cf, f708d80, 41eca74, d7a9e68, 236fb8b), relicensed to MIT OR
//! Apache-2.0 by the author.

/// RDP scancode of Num Lock (non-extended).
pub(super) const SCANCODE_NUM_LOCK: u8 = 0x45;

/// RDP scancode of Left Ctrl (non-extended; Right Ctrl is the extended one).
const SCANCODE_LEFT_CTRL: u8 = 0x1D;

/// The client's Num Lock state, as Windows keeps it. macOS has no Num Lock:
/// the server keeps it so that the keypad keys move the cursor when it is
/// off, as on Windows (`scancodes::keypad_navigation_vk`).
#[derive(Debug)]
pub(super) struct NumLock {
    on: bool,
    /// A Num Lock key-down was posted and its key-up was not.
    key_down: bool,
    /// The client's Left Ctrl (scancode 0x1D, not extended) is down.
    left_ctrl: bool,
}

impl Default for NumLock {
    /// On until the client's Synchronize says otherwise: keypad keys type
    /// digits, as before this state existed.
    fn default() -> Self {
        Self {
            on: true,
            key_down: false,
            left_ctrl: false,
        }
    }
}

impl NumLock {
    pub(super) fn is_on(&self) -> bool {
        self.on
    }

    /// Every scancode event, so that Pause can be told from Num Lock.
    pub(super) fn note_scancode(&mut self, scancode: u8, extended: bool, pressed: bool) {
        if scancode == SCANCODE_LEFT_CTRL && !extended {
            self.left_ctrl = pressed;
        }
    }

    /// Scancode 0x45 (not extended), pressed or released. A Num Lock press
    /// toggles the state. Returns whether to post the event (as Num Lock): a
    /// release is posted only after a posted press.
    ///
    /// mstsc and FreeRDP send Pause as Left Ctrl with KBDFLAGS_EXTENDED1, then
    /// 0x45 (MS-RDPBCGR 2.2.8.1.1.3.1.1.1). The server library passes on only
    /// the extended flag, so Pause looks like Ctrl+Num Lock, which Windows
    /// also reads as Pause. Pause is not posted (macOS has no Pause key) and
    /// does not toggle Num Lock.
    pub(super) fn key(&mut self, pressed: bool) -> bool {
        if !pressed {
            return std::mem::take(&mut self.key_down);
        }
        if self.left_ctrl {
            return false;
        }
        self.on = !self.on;
        self.key_down = true;
        true
    }

    /// RDP Synchronize: the client's Num Lock state; all keys are up.
    pub(super) fn synchronize(&mut self, on: bool) {
        self.on = on;
        self.left_ctrl = false;
    }

    /// The posted Num Lock press was released by other means (connection
    /// end): no key-up is due any more, and no client key is down. The state
    /// stays.
    pub(super) fn forget_key(&mut self) {
        self.key_down = false;
        self.left_ctrl = false;
    }
}

/// The macOS keycode posted for each client key that is down, by
/// (scancode, extended). The key-up posts the same keycode, also when the
/// state that chose it changed in between: keypad 4 down with Num Lock off
/// posts Left Arrow; if Num Lock goes on before the release, the release must
/// be Left Arrow too, not keypad 4 (the arrow would stay down).
#[derive(Debug, Default)]
pub(super) struct PostedKeys {
    down: std::collections::HashMap<(u8, bool), u16>,
}

impl PostedKeys {
    /// The keycode to post for this scancode event; `vk` is the one the
    /// current state gives.
    pub(super) fn vk_for(&mut self, key: (u8, bool), pressed: bool, vk: u16) -> u16 {
        if pressed {
            *self.down.entry(key).or_insert(vk)
        } else {
            self.down.remove(&key).unwrap_or(vk)
        }
    }

    /// All keys are up (connection end).
    pub(super) fn clear(&mut self) {
        self.down.clear();
    }
}

/// What to post for an RDP Unicode keyboard event.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum UnicodeAction {
    /// Key-down carrying these UTF-16 units (one character).
    Down(Vec<u16>),
    /// Key-up carrying these UTF-16 units (the character of the down).
    Up(Vec<u16>),
    /// Nothing to post.
    None,
}

/// RDP Unicode key events (MS-RDPBCGR TS_UNICODE_KEYBOARD_EVENT): one UTF-16
/// unit per event, a down and an up for each. A character outside the BMP
/// (emoji) comes as a high and a low surrogate; a lone surrogate is not valid
/// text, so the pair is joined into one key-down. Also counts the downs
/// posted without their up, for the connection end.
#[derive(Debug, Default)]
pub(super) struct UnicodeKeys {
    high: Option<u16>,
    /// The joined pair posted as a key-down: its low surrogate release posts
    /// the key-up.
    pair_down: Option<[u16; 2]>,
    /// Key-downs posted and not released yet.
    down: u32,
}

impl UnicodeKeys {
    pub(super) fn feed(&mut self, unit: u16, pressed: bool) -> UnicodeAction {
        let action = match unit {
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
                    self.pair_down = Some([high, unit]);
                    UnicodeAction::Down(vec![high, unit])
                }
                // A low surrogate without a high one is dropped.
                (true, None) => UnicodeAction::None,
                (false, _) => match self.pair_down.take() {
                    Some(pair) => UnicodeAction::Up(pair.to_vec()),
                    None => UnicodeAction::None,
                },
            },
            _ => {
                if pressed {
                    self.high = None;
                    UnicodeAction::Down(vec![unit])
                } else {
                    UnicodeAction::Up(vec![unit])
                }
            }
        };
        match action {
            UnicodeAction::Down(_) => self.down += 1,
            UnicodeAction::Up(_) => self.down = self.down.saturating_sub(1),
            UnicodeAction::None => {}
        }
        action
    }

    /// Forget every Unicode key; returns whether one was down (then one
    /// key-up is due: all Unicode downs use the same keycode).
    pub(super) fn take_down(&mut self) -> bool {
        std::mem::take(self).down > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bmp_character_is_posted_as_is() {
        let mut u = UnicodeKeys::default();
        assert_eq!(u.feed(0x00E9, true), UnicodeAction::Down(vec![0x00E9]));
        assert_eq!(u.feed(0x00E9, false), UnicodeAction::Up(vec![0x00E9]));
        assert!(!u.take_down());
    }

    #[test]
    fn surrogate_pair_becomes_one_character() {
        // U+1F600 = D83D DE00; mstsc order: high down, high up, low down, low up
        let mut u = UnicodeKeys::default();
        assert_eq!(u.feed(0xD83D, true), UnicodeAction::None);
        assert_eq!(u.feed(0xD83D, false), UnicodeAction::None);
        assert_eq!(
            u.feed(0xDE00, true),
            UnicodeAction::Down(vec![0xD83D, 0xDE00])
        );
        assert_eq!(
            u.feed(0xDE00, false),
            UnicodeAction::Up(vec![0xD83D, 0xDE00])
        );
        assert!(!u.take_down());
    }

    #[test]
    fn surrogate_pair_with_nested_releases() {
        let mut u = UnicodeKeys::default();
        assert_eq!(u.feed(0xD83D, true), UnicodeAction::None);
        assert_eq!(
            u.feed(0xDE00, true),
            UnicodeAction::Down(vec![0xD83D, 0xDE00])
        );
        assert_eq!(
            u.feed(0xDE00, false),
            UnicodeAction::Up(vec![0xD83D, 0xDE00])
        );
        assert_eq!(u.feed(0xD83D, false), UnicodeAction::None);
    }

    #[test]
    fn lone_surrogates_are_dropped() {
        let mut u = UnicodeKeys::default();
        assert_eq!(u.feed(0xDE00, true), UnicodeAction::None);
        assert_eq!(u.feed(0xDE00, false), UnicodeAction::None);
        // a high surrogate followed by a BMP character is dropped too
        assert_eq!(u.feed(0xD83D, true), UnicodeAction::None);
        assert_eq!(u.feed(0x0041, true), UnicodeAction::Down(vec![0x0041]));
        assert_eq!(u.feed(0xDE00, true), UnicodeAction::None);
    }

    #[test]
    fn a_unicode_key_stays_down_until_all_are_up() {
        let mut u = UnicodeKeys::default();
        u.feed(0x00E9, true);
        u.feed(0x00FC, true);
        u.feed(0x00E9, false);
        assert!(u.take_down(), "one is still down");
        assert!(!u.take_down(), "take_down forgets it");
        // Both up and a stray up: nothing is down.
        u.feed(0x00E9, true);
        u.feed(0x00E9, false);
        u.feed(0x00E9, false);
        assert!(!u.take_down());
    }

    #[test]
    fn key_up_posts_the_keycode_of_its_key_down() {
        let mut p = PostedKeys::default();
        // Keypad 4 (0x4B) down with Num Lock off: Left Arrow.
        assert_eq!(p.vk_for((0x4B, false), true, 0x7B), 0x7B);
        // Client key repeat after Num Lock went on: still Left Arrow.
        assert_eq!(p.vk_for((0x4B, false), true, 0x56), 0x7B);
        // The release with Num Lock on is also Left Arrow, not keypad 4.
        assert_eq!(p.vk_for((0x4B, false), false, 0x56), 0x7B);
        // The next press uses the state again.
        assert_eq!(p.vk_for((0x4B, false), true, 0x56), 0x56);
        // The extended scancode is another key.
        assert_eq!(p.vk_for((0x4B, true), false, 0x7B), 0x7B);
        p.clear();
        assert_eq!(p.vk_for((0x4B, false), false, 0x7B), 0x7B);
    }

    #[test]
    fn num_lock_starts_on_and_follows_synchronize() {
        let mut n = NumLock::default();
        assert!(n.is_on());
        n.synchronize(false);
        assert!(!n.is_on());
        n.synchronize(true);
        assert!(n.is_on());
    }

    #[test]
    fn num_lock_key_toggles_on_press_only() {
        let mut n = NumLock::default();
        assert!(n.key(true));
        assert!(!n.is_on());
        assert!(n.key(false));
        assert!(!n.is_on());
        assert!(n.key(true));
        assert!(n.is_on());
    }

    #[test]
    fn a_release_without_a_posted_press_is_not_posted() {
        let mut n = NumLock::default();
        assert!(!n.key(false));
        assert!(n.key(true));
        n.forget_key();
        assert!(!n.key(false), "the connection end already released it");
        assert!(!n.is_on(), "the state stays");
    }

    /// mstsc Pause: Left Ctrl (EXTENDED1) down, 0x45 down, Ctrl up, 0x45 up.
    #[test]
    fn pause_is_not_num_lock() {
        let mut n = NumLock::default();
        n.note_scancode(0x1D, false, true);
        assert!(!n.key(true), "Pause is not posted");
        n.note_scancode(0x1D, false, false);
        assert!(!n.key(false));
        assert!(n.is_on(), "Pause does not toggle Num Lock");
        // The next Num Lock press is Num Lock again.
        assert!(n.key(true));
        assert!(!n.is_on());
    }

    #[test]
    fn right_ctrl_is_not_the_pause_prefix() {
        let mut n = NumLock::default();
        n.note_scancode(0x1D, true, true);
        assert!(n.key(true));
        assert!(!n.is_on());
    }

    #[test]
    fn synchronize_forgets_a_held_left_ctrl() {
        let mut n = NumLock::default();
        n.note_scancode(0x1D, false, true);
        n.synchronize(true);
        assert!(n.key(true), "Ctrl went up with the Synchronize");
    }
}
