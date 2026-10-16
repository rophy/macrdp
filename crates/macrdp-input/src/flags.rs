//! Modifier flags for injected events.
//!
//! `CGEventCreateKeyboardEvent` and `CGEventCreateMouseEvent` take their flags
//! from the HID system state, and an event posted at the HID tap writes its
//! flags back into that state. So a key that macOS marks with fn/numpad/help
//! (arrows, Home, F1, Insert) leaves those flags in the state, and every later
//! letter is created as fn+letter (a shortcut, not text). Also, in a burst the
//! state can lag behind the Shift key events, which gives the wrong case.
//!
//! The injector therefore keeps the client's modifier state itself and sets
//! the flags of every event it posts.

/// `kCGEventFlagMaskAlphaShift` (Caps Lock)
pub const ALPHA_SHIFT: u64 = 0x0001_0000;
/// `kCGEventFlagMaskShift`
pub const SHIFT: u64 = 0x0002_0000;
/// `kCGEventFlagMaskControl`
pub const CONTROL: u64 = 0x0004_0000;
/// `kCGEventFlagMaskAlternate` (Option)
pub const ALTERNATE: u64 = 0x0008_0000;
/// `kCGEventFlagMaskCommand`
pub const COMMAND: u64 = 0x0010_0000;
/// `kCGEventFlagMaskNumericPad`
pub const NUMERIC_PAD: u64 = 0x0020_0000;
/// `kCGEventFlagMaskHelp`
pub const HELP: u64 = 0x0040_0000;
/// `kCGEventFlagMaskSecondaryFn`
pub const SECONDARY_FN: u64 = 0x0080_0000;

// Device-dependent bits for the left/right modifier keys (IOLLEvent.h NX_DEVICE*KEYMASK).
const DEV_LCTL: u64 = 0x0001;
const DEV_LSHIFT: u64 = 0x0002;
const DEV_RSHIFT: u64 = 0x0004;
const DEV_LCMD: u64 = 0x0008;
const DEV_RCMD: u64 = 0x0010;
const DEV_LALT: u64 = 0x0020;
const DEV_RALT: u64 = 0x0040;
const DEV_RCTL: u64 = 0x2000;

/// The bits that the injector sets itself; all other bits of a created event are kept.
pub const MANAGED: u64 = ALPHA_SHIFT
    | SHIFT
    | CONTROL
    | ALTERNATE
    | COMMAND
    | NUMERIC_PAD
    | HELP
    | SECONDARY_FN
    | DEV_LCTL
    | DEV_LSHIFT
    | DEV_RSHIFT
    | DEV_LCMD
    | DEV_RCMD
    | DEV_LALT
    | DEV_RALT
    | DEV_RCTL;

/// macOS keycode of Caps Lock (`kVK_CapsLock`)
pub const KEYCODE_CAPS_LOCK: u16 = 0x39;

/// (macOS keycode, generic flag, device-dependent flag) of each modifier key.
const MODIFIER_KEYS: [(u16, u64, u64); 8] = [
    (0x38, SHIFT, DEV_LSHIFT),   // kVK_Shift
    (0x3C, SHIFT, DEV_RSHIFT),   // kVK_RightShift
    (0x3B, CONTROL, DEV_LCTL),   // kVK_Control
    (0x3E, CONTROL, DEV_RCTL),   // kVK_RightControl
    (0x3A, ALTERNATE, DEV_LALT), // kVK_Option
    (0x3D, ALTERNATE, DEV_RALT), // kVK_RightOption
    (0x37, COMMAND, DEV_LCMD),   // kVK_Command
    (0x36, COMMAND, DEV_RCMD),   // kVK_RightCommand
];

/// Index of Left Control in `MODIFIER_KEYS`.
const LEFT_CONTROL: usize = 2;

/// Flags that a key gets by itself (no modifier held). The table is what
/// `CGEventCreateKeyboardEvent` sets for keycodes 0-127 from a clean state
/// (macOS 15): arrows get numpad+fn, the navigation keys and keypad Clear get
/// fn, Help gets help+fn, other keypad keys get numpad.
///
/// Exception: F1-F20 get no fn flag, although macOS gives them one. The
/// system shortcuts on F-keys are defined with fn (Show Desktop is fn+F11),
/// so an F-key with fn goes to the system and not to the app. A remote user
/// expects the F-key in the app, as on Windows.
pub fn intrinsic_flags(keycode: u16) -> u64 {
    match keycode {
        // arrows
        0x7B..=0x7E => NUMERIC_PAD | SECONDARY_FN,
        // Help (Insert)
        0x72 => HELP | SECONDARY_FN,
        // Home, PgUp, Fwd Delete, End, PgDn
        0x73 | 0x74 | 0x75 | 0x77 | 0x79 => SECONDARY_FN,
        // fn, keypad Clear
        0x3F | 0x47 => SECONDARY_FN,
        // keypad . * + / Enter - = 0-9
        0x41 | 0x43 | 0x45 | 0x4B | 0x4C | 0x4E | 0x51 | 0x52..=0x59 | 0x5B | 0x5C => NUMERIC_PAD,
        _ => 0,
    }
}

/// Keep the bits of a created event that the injector does not manage and set the rest.
pub fn merge(created: u64, ours: u64) -> u64 {
    (created & !MANAGED) | (ours & MANAGED)
}

/// Modifier and lock state of the client keyboard, as seen through the RDP input events.
#[derive(Debug, Clone)]
pub struct ModifierState {
    /// Bit i = MODIFIER_KEYS[i] is down.
    down: u8,
    caps_lock: bool,
    num_lock: bool,
    /// The scancode 0x45 that is down is the Pause key
    pause_down: bool,
}

impl Default for ModifierState {
    fn default() -> Self {
        // Num Lock on until the client sends its state: keypad keys type digits.
        Self {
            down: 0,
            caps_lock: false,
            num_lock: true,
            pause_down: false,
        }
    }
}

impl ModifierState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a key press or release (macOS keycode). Caps Lock toggles on press.
    pub fn key(&mut self, keycode: u16, pressed: bool) {
        if keycode == KEYCODE_CAPS_LOCK {
            if pressed {
                self.caps_lock = !self.caps_lock;
            }
            return;
        }
        if let Some(i) = MODIFIER_KEYS.iter().position(|(kc, _, _)| *kc == keycode) {
            if pressed {
                self.down |= 1 << i;
            } else {
                self.down &= !(1 << i);
            }
        }
    }

    /// RDP Synchronize event: set the lock states and release all keys
    /// (MS-RDPBCGR 2.2.8.1.1.3.1.1.5: the server sets all keys up; the client
    /// then sends a key down for each key that is still down). Returns the
    /// modifier keycodes that were down: they need a key up.
    pub fn synchronize(&mut self, caps_lock: bool, num_lock: bool) -> Vec<u16> {
        self.caps_lock = caps_lock;
        self.num_lock = num_lock;
        self.pause_down = false;
        let held = MODIFIER_KEYS
            .iter()
            .enumerate()
            .filter(|(i, _)| self.down & (1 << i) != 0)
            .map(|(_, (kc, _, _))| *kc)
            .collect();
        self.down = 0;
        held
    }

    /// Scancode 0x45 without the extended flag, pressed or released. Returns
    /// whether to post it (as Num Lock).
    ///
    /// mstsc and FreeRDP send Pause as Left Ctrl with KBDFLAGS_EXTENDED1, then
    /// 0x45 (MS-RDPBCGR 2.2.8.1.1.3.1.1.1). The server library gives only the
    /// extended flag, so that Pause looks like Ctrl+Num Lock, which Windows
    /// also reads as Pause. Pause is not posted (macOS has no Pause key) and
    /// does not toggle Num Lock.
    pub fn num_lock_scancode(&mut self, pressed: bool) -> bool {
        if !pressed {
            return !std::mem::take(&mut self.pause_down);
        }
        if self.down & (1 << LEFT_CONTROL) != 0 {
            self.pause_down = true;
            return false;
        }
        self.num_lock = !self.num_lock;
        true
    }

    pub fn num_lock(&self) -> bool {
        self.num_lock
    }

    /// Flags of the held modifiers and Caps Lock (for mouse events and Unicode input).
    pub fn flags(&self) -> u64 {
        let mut f = if self.caps_lock { ALPHA_SHIFT } else { 0 };
        for (i, (_, generic, device)) in MODIFIER_KEYS.iter().enumerate() {
            if self.down & (1 << i) != 0 {
                f |= generic | device;
            }
        }
        f
    }

    /// Flags for an event of `keycode`: held modifiers plus the key's own flags.
    pub fn key_flags(&self, keycode: u16) -> u64 {
        self.flags() | intrinsic_flags(keycode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_after_an_arrow_has_no_fn_flag() {
        let mut s = ModifierState::new();
        s.key(0x7C, true);
        assert_eq!(s.key_flags(0x7C), NUMERIC_PAD | SECONDARY_FN);
        s.key(0x7C, false);
        assert_eq!(s.key_flags(0x00), 0);
    }

    #[test]
    fn navigation_and_function_keys_have_their_own_flags() {
        let s = ModifierState::new();
        assert_eq!(s.key_flags(0x73), SECONDARY_FN); // Home
        assert_eq!(s.key_flags(0x75), SECONDARY_FN); // Fwd Delete
        assert_eq!(s.key_flags(0x72), HELP | SECONDARY_FN); // Help
        assert_eq!(s.key_flags(0x53), NUMERIC_PAD); // keypad 1
        assert_eq!(s.key_flags(0x4C), NUMERIC_PAD); // keypad Enter
    }

    #[test]
    fn function_keys_have_no_fn_flag() {
        // fn+F11 is Show Desktop; the app must get F1-F20 as plain keys.
        let mut s = ModifierState::new();
        for kc in [0x7A, 0x60, 0x67, 0x6F, 0x69, 0x5A] {
            assert_eq!(s.key_flags(kc), 0, "keycode {kc:#x}");
        }
        s.key(0x3B, true); // Control
        assert_eq!(s.key_flags(0x78), CONTROL | DEV_LCTL); // Ctrl+F2
    }

    #[test]
    fn shift_is_set_while_held_and_cleared_on_release() {
        let mut s = ModifierState::new();
        s.key(0x38, true);
        assert_eq!(s.key_flags(0x00), SHIFT | DEV_LSHIFT);
        s.key(0x38, false);
        assert_eq!(s.key_flags(0x00), 0);
    }

    #[test]
    fn both_shift_keys_must_be_released() {
        let mut s = ModifierState::new();
        s.key(0x38, true);
        s.key(0x3C, true);
        s.key(0x38, false);
        assert_eq!(s.flags(), SHIFT | DEV_RSHIFT);
        s.key(0x3C, false);
        assert_eq!(s.flags(), 0);
    }

    #[test]
    fn modifiers_combine() {
        let mut s = ModifierState::new();
        s.key(0x3B, true); // Control
        s.key(0x37, true); // Command
        s.key(0x3A, true); // Option
        assert_eq!(
            s.flags(),
            CONTROL | COMMAND | ALTERNATE | DEV_LCTL | DEV_LCMD | DEV_LALT
        );
    }

    #[test]
    fn caps_lock_toggles_on_press_only() {
        let mut s = ModifierState::new();
        s.key(KEYCODE_CAPS_LOCK, true);
        s.key(KEYCODE_CAPS_LOCK, false);
        assert_eq!(s.flags(), ALPHA_SHIFT);
        s.key(KEYCODE_CAPS_LOCK, true);
        s.key(KEYCODE_CAPS_LOCK, false);
        assert_eq!(s.flags(), 0);
    }

    #[test]
    fn synchronize_sets_the_locks() {
        let mut s = ModifierState::new();
        assert!(s.num_lock());
        assert!(s.synchronize(true, false).is_empty());
        assert_eq!(s.flags(), ALPHA_SHIFT);
        assert!(!s.num_lock());
    }

    #[test]
    fn synchronize_releases_held_modifiers() {
        // Alt (Command) went down, its release went to another window.
        let mut s = ModifierState::new();
        s.key(0x37, true);
        s.key(0x38, true);
        assert_eq!(s.synchronize(false, false), vec![0x38, 0x37]);
        assert_eq!(s.key_flags(0x05), 0);
        assert!(s.synchronize(false, false).is_empty());
    }

    #[test]
    fn num_lock_key_toggles() {
        let mut s = ModifierState::new();
        assert!(s.num_lock_scancode(true));
        assert!(!s.num_lock());
        assert!(s.num_lock_scancode(false));
        assert!(s.num_lock_scancode(true));
        assert!(s.num_lock());
    }

    #[test]
    fn pause_is_not_num_lock() {
        // mstsc Pause: Ctrl (EXTENDED1) down, 0x45 down, Ctrl up, 0x45 up
        let mut s = ModifierState::new();
        s.key(0x3B, true);
        assert!(!s.num_lock_scancode(true));
        s.key(0x3B, false);
        assert!(!s.num_lock_scancode(false));
        assert!(s.num_lock());
        // The next Num Lock press is Num Lock again.
        assert!(s.num_lock_scancode(true));
        assert!(!s.num_lock());
    }

    #[test]
    fn merge_replaces_the_state_flags_and_keeps_other_bits() {
        // 0x2000_0000 and 0x100 (non-coalesced) are not managed; fn/numpad/help come from a stale state.
        let created = 0x2000_0000 | 0x100 | NUMERIC_PAD | SECONDARY_FN | HELP;
        assert_eq!(
            merge(created, SHIFT | DEV_LSHIFT),
            0x2000_0100 | SHIFT | DEV_LSHIFT
        );
    }
}
