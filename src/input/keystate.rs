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
    ///
    /// A held key repeats its key-down (typematic) until the key-up: only the
    /// first press toggles, as on Windows; a repeat is posted as a repeat.
    pub(super) fn key(&mut self, pressed: bool) -> bool {
        if !pressed {
            return std::mem::take(&mut self.key_down);
        }
        if self.key_down {
            return true;
        }
        if self.left_ctrl {
            return false;
        }
        self.on = !self.on;
        self.key_down = true;
        true
    }

    /// RDP Synchronize: the client's Num Lock state. Whether the keys went
    /// up with it is decided elsewhere (`FocusIn`); if they did, `forget_key`
    /// follows.
    pub(super) fn synchronize(&mut self, on: bool) {
        self.on = on;
    }

    /// The posted Num Lock press was released by other means (connection
    /// end, focus-in Synchronize): no key-up is due any more, and no client
    /// key is down. The state stays.
    pub(super) fn forget_key(&mut self) {
        self.key_down = false;
        self.left_ctrl = false;
    }
}

/// RDP scancode of Tab (non-extended).
const SCANCODE_TAB: u8 = 0x0F;

/// How long after its Tab key-up a Synchronize still counts as a focus-in.
/// Measured at the server (2026-10-21): mstsc (Windows 10) 0.1-1 ms, FreeRDP
/// 3.30 0.2 ms; the client sends the two together. A typed Tab followed by a
/// key that makes guacd change a lock is a keystroke later (157 ms measured
/// with a scripted 150 ms gap); the margin is for a split network packet.
const FOCUS_IN_WINDOW_MS: u64 = 50;

/// What a trigger releases on the Mac.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Release {
    /// Keys, modifiers and mouse buttons: a connection ends or starts.
    All,
    /// Keys and modifiers, not the mouse buttons: a focus-in Synchronize.
    KeysOnly,
    /// Nothing: a Synchronize that only changes a lock state.
    Nothing,
}

impl Release {
    pub(super) fn keys(self) -> bool {
        self != Release::Nothing
    }

    /// Only at a connection edge. mstsc sends the button-down of the click
    /// that gives it the focus BEFORE the Synchronize (measured 2026-10-21),
    /// so releasing buttons on a Synchronize ended that click at once.
    pub(super) fn buttons(self) -> bool {
        self == Release::All
    }
}

/// Tells a focus-in Synchronize from one that only changes a lock state.
///
/// MS-RDPBCGR 2.2.8.1.1.3.1.1.5 says a Synchronize sets all keys up, and the
/// client then sends key-downs for the keys still down. Clients do not all
/// work that way:
/// - mstsc and FreeRDP send it when their window gets the focus back, as Tab
///   up, Synchronize, Tab up (mstsc: three times, with the modifier key-ups
///   after each Synchronize and a key-down for a modifier still held). mstsc
///   does not send the key-ups of keys released in another window, so those
///   keys must be released here.
/// - guacd also sends it in the middle of typing, when the next key needs
///   another lock state (keypad digits need Num Lock on; guacamole-server
///   src/protocols/rdp/keyboard.c `guac_rdp_keyboard_update_locks`), without
///   a Tab up and without sending the held keys again. Releasing there let go
///   of a Ctrl that the user still held.
///
/// So the keys are released only on a Synchronize that follows a Tab key-up
/// within `FOCUS_IN_WINDOW_MS`, with no key-down in between.
#[derive(Debug, Default)]
pub(super) struct FocusIn {
    /// When the last Tab key-up came, if no key went down after it.
    tab_up_ms: Option<u64>,
}

impl FocusIn {
    /// Every scancode key event.
    pub(super) fn note_key(&mut self, scancode: u8, extended: bool, pressed: bool, now_ms: u64) {
        if pressed {
            self.tab_up_ms = None;
        } else if scancode == SCANCODE_TAB && !extended {
            self.tab_up_ms = Some(now_ms);
        }
    }

    /// Every Unicode key event.
    pub(super) fn note_unicode(&mut self, pressed: bool) {
        if pressed {
            self.tab_up_ms = None;
        }
    }

    /// A Synchronize: what to release.
    pub(super) fn synchronize(&mut self, now_ms: u64) -> Release {
        match self.tab_up_ms.take() {
            Some(t) if now_ms.saturating_sub(t) <= FOCUS_IN_WINDOW_MS => Release::KeysOnly,
            _ => Release::Nothing,
        }
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

    // --- Synchronize: focus-in or lock change (sequences measured 2026-10-21,
    // see FocusIn) ---

    /// FreeRDP and mstsc on focus-in: Tab up, Synchronize, Tab up.
    #[test]
    fn focus_in_synchronize_releases_the_keys() {
        let mut f = FocusIn::default();
        f.note_key(0x25, false, true, 1_000); // K down, then the focus goes
        f.note_key(0x0F, false, false, 9_000);
        assert_eq!(f.synchronize(9_000), Release::KeysOnly);
        f.note_key(0x0F, false, false, 9_000);
    }

    /// guacd: a Synchronize in the middle of typing when a key needs another
    /// lock state (Ctrl held, keypad 1 with Num Lock off): no Tab up before it.
    #[test]
    fn lock_synchronize_while_typing_keeps_the_keys() {
        let mut f = FocusIn::default();
        assert_eq!(
            f.synchronize(500),
            Release::Nothing,
            "first key of the session"
        );
        f.note_key(0x1D, false, true, 500); // Ctrl down
        assert_eq!(f.synchronize(800), Release::Nothing);
    }

    /// A Tab typed by the user, then a lock Synchronize at the next key: the
    /// Tab up is too old to be a focus-in, or a key went down after it.
    #[test]
    fn a_typed_tab_is_not_a_focus_in() {
        let mut f = FocusIn::default();
        f.note_key(0x1D, false, true, 0);
        f.note_key(0x0F, false, true, 100);
        f.note_key(0x0F, false, false, 180);
        assert_eq!(f.synchronize(400), Release::Nothing, "too late");

        f.note_key(0x0F, false, true, 1_000);
        f.note_key(0x0F, false, false, 1_080);
        f.note_key(0x4F, false, true, 1_090);
        assert_eq!(
            f.synchronize(1_090),
            Release::Nothing,
            "a key went down after the Tab up"
        );
    }

    /// Key-ups between the Tab up and the Synchronize (mstsc re-sends the
    /// modifier ups) keep the pattern; a Unicode key-down breaks it.
    #[test]
    fn key_ups_keep_the_focus_in_pattern() {
        let mut f = FocusIn::default();
        f.note_key(0x0F, false, false, 50);
        f.note_key(0x2A, false, false, 50);
        assert_eq!(f.synchronize(51), Release::KeysOnly);
        f.note_key(0x0F, false, false, 60);
        f.note_unicode(true);
        assert_eq!(f.synchronize(60), Release::Nothing);
    }

    /// Each Synchronize uses up its Tab up.
    #[test]
    fn one_tab_up_is_one_focus_in() {
        let mut f = FocusIn::default();
        f.note_key(0x0F, false, false, 0);
        assert_eq!(f.synchronize(0), Release::KeysOnly);
        assert_eq!(f.synchronize(1), Release::Nothing);
    }

    /// What each trigger releases. Buttons go only at a connection edge: mstsc
    /// sends the button-down of the click that gives it the focus BEFORE the
    /// Synchronize, so releasing buttons there ended that click at once.
    #[test]
    fn synchronize_never_releases_mouse_buttons() {
        assert!(Release::All.keys() && Release::All.buttons());
        assert!(Release::KeysOnly.keys());
        assert!(!Release::KeysOnly.buttons());
        assert!(!Release::Nothing.keys() && !Release::Nothing.buttons());
        let mut f = FocusIn::default();
        f.note_key(0x0F, false, false, 0);
        assert!(!f.synchronize(0).buttons());
        assert!(!f.synchronize(0).buttons());
    }

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

    /// A held Num Lock: the client repeats the key-down (typematic) until the
    /// key-up. Only the first press toggles, as on Windows.
    #[test]
    fn num_lock_auto_repeat_toggles_once() {
        let mut n = NumLock::default();
        assert!(n.key(true));
        assert!(!n.is_on());
        for _ in 0..5 {
            assert!(n.key(true), "a repeat is posted (the key is down)");
            assert!(!n.is_on(), "a repeat does not toggle");
        }
        assert!(n.key(false));
        assert!(!n.is_on());
        assert!(n.key(true), "a new press toggles again");
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
    fn released_keys_forget_a_held_left_ctrl() {
        let mut n = NumLock::default();
        n.note_scancode(0x1D, false, true);
        n.forget_key();
        assert!(n.key(true), "Ctrl went up with the focus-in");
    }

    /// guacd's lock Synchronize keeps the keys: a Left Ctrl still held still
    /// makes 0x45 Pause.
    #[test]
    fn lock_synchronize_keeps_a_held_left_ctrl() {
        let mut n = NumLock::default();
        n.note_scancode(0x1D, false, true);
        n.synchronize(false);
        assert!(!n.key(true), "still Pause");
        assert!(!n.is_on());
    }
}
