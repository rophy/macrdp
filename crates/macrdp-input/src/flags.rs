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

/// Flags that macOS gives a key by itself (no modifier held). The table is
/// what `CGEventCreateKeyboardEvent` sets for keycodes 0-127 from a clean
/// state (macOS 15): arrows get numpad+fn, the navigation and function keys
/// and keypad Clear get fn, Help gets help+fn, other keypad keys get numpad.
pub fn intrinsic_flags(keycode: u16) -> u64 {
    match keycode {
        // arrows
        0x7B..=0x7E => NUMERIC_PAD | SECONDARY_FN,
        // Help (Insert)
        0x72 => HELP | SECONDARY_FN,
        // Home, PgUp, Fwd Delete, End, PgDn
        0x73 | 0x74 | 0x75 | 0x77 | 0x79 => SECONDARY_FN,
        // F1-F20
        0x7A | 0x78 | 0x63 | 0x76 | 0x60 | 0x61 | 0x62 | 0x64 | 0x65 | 0x6D | 0x67 | 0x6F
        | 0x69 | 0x6B | 0x71 | 0x6A | 0x40 | 0x4F | 0x50 | 0x5A => SECONDARY_FN,
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
}

impl Default for ModifierState {
    fn default() -> Self {
        // Num Lock on until the client sends its state: keypad keys type digits.
        Self {
            down: 0,
            caps_lock: false,
            num_lock: true,
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

    /// Set the lock states (RDP Synchronize event).
    pub fn set_locks(&mut self, caps_lock: bool, num_lock: bool) {
        self.caps_lock = caps_lock;
        self.num_lock = num_lock;
    }

    pub fn toggle_num_lock(&mut self) {
        self.num_lock = !self.num_lock;
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
        assert_eq!(s.key_flags(0x7A), SECONDARY_FN); // F1
        assert_eq!(s.key_flags(0x6F), SECONDARY_FN); // F12
        assert_eq!(s.key_flags(0x72), HELP | SECONDARY_FN); // Help
        assert_eq!(s.key_flags(0x53), NUMERIC_PAD); // keypad 1
        assert_eq!(s.key_flags(0x4C), NUMERIC_PAD); // keypad Enter
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
        s.set_locks(true, false);
        assert_eq!(s.flags(), ALPHA_SHIFT);
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
