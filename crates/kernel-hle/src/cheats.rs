//! Cheats for Skyrim: what the PC version's console is mostly used for, done with the title's
//! own functions.
//!
//! The Xbox 360 build has no console, and its Papyrus `Debug` natives (`SetGodMode`,
//! `ToggleCollisions`...) are empty functions. The rest of Papyrus is there, though, and each
//! native's registration names the function that does the work, which is how the addresses
//! below were found (`Game.GetPlayer`, `Game.GetForm`, `Game.AddPerkPoints`,
//! `Game.IncrementSkill`, `ObjectReference.AddItem`, `Actor.RestoreActorValue`,
//! `Actor.ModActorValue`, `Actor.SetActorValue`).
//!
//! A host asks for a cheat with [`request`] (or [`set_god_mode`]); requests are carried out on
//! the title's main thread where it ends a frame's job list (`skyrt_cheats_tick`, called by
//! guest-sys's override of the job list's stop function), when no job thread is running game
//! code. God mode is a check in guest-sys's override of the function every change of an actor
//! value goes through.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use runtime_core::PPCContext;

use crate::Kernel;

/// Skyrim (title 425307E6, title update 0x00000A02): the addresses below are this build's.
pub const TITLE: (u32, u32) = (0x4253_07E6, 0x0000_0A02);

/// `PlayerCharacter*` (what `Game.GetPlayer` returns).
pub const PLAYER: u32 = 0x8385_DEDC;
/// `TESForm* LookupFormByID(u32 id)`.
const LOOKUP_FORM: u32 = 0x8238_9738;
/// `(TESObjectREFR* to, TESForm* item, u32 count, bool silent, u32 stack id, VM* vm)`: what
/// `ObjectReference.AddItem` ends in. The last two are only used to report an item that can't
/// be added.
const ADD_ITEM: u32 = 0x829F_5AB8;
/// `(Actor*, u32 value id, f32 amount)`: takes that much off the damage done to the value.
const RESTORE_VALUE: u32 = 0x8242_C580;
/// `(PlayerSkills*, u32 value id, u32 levels)`: what `Game.IncrementSkill` ends in.
const INCREMENT_SKILL: u32 = 0x8280_F350;
/// `u32 (Actor*)`: the gold an actor carries (`Actor.GetGoldAmount`).
const GOLD_AMOUNT: u32 = 0x8274_0A88;

/// `TESObjectREFR::parentCell`: null while no game is loaded.
const PARENT_CELL: u32 = 60;
/// The `ActorValueOwner` inside an `Actor`. Its virtual functions: 1 = get (value id) -> f32,
/// 6 = change (modifier: 0 permanent, 1 temporary, 2 damage; value id; f32 amount),
/// 7 = set (value id, f32).
pub const VALUE_OWNER: u32 = 92;
const GET_VALUE: u32 = 1;
const MOD_VALUE: u32 = 6;
const SET_VALUE: u32 = 7;
/// `PlayerCharacter`: perk points to spend (a byte) and the `PlayerSkills*`.
const PERK_POINTS: u32 = 1757;
const SKILLS: u32 = 1552;

/// Actor value ids. The skills are 6 (one-handed) to 23 (enchanting).
const SKILL_VALUES: std::ops::Range<u32> = 6..24;
pub const HEALTH: u32 = 24;
const MAGICKA: u32 = 25;
const STAMINA: u32 = 26;
const SPEED_MULT: u32 = 30;
const CARRY_WEIGHT: u32 = 32;

/// Form ids.
const GOLD: u32 = 0xF;
const LOCKPICK: u32 = 0xA;

/// The guest thread the title's main loop runs on.
const MAIN_THREAD: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cheat {
    Gold(u32),
    Lockpicks(u32),
    /// Health, magicka and stamina back to full.
    Heal,
    /// More (or less) weight the player can carry, for good (as `player.modav carryweight`).
    CarryWeight(f32),
    PerkPoints(u8),
    /// Every skill up by this many levels (the character's level follows, as when training).
    SkillLevels(u32),
    /// The player's movement speed in percent (100 is the game's).
    RunSpeed(f32),
    /// Only logs the player's state (for tests).
    Status,
}

/// Read by the generated overrides: requests are waiting; the player takes no damage.
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static skyrt_cheats_pending: AtomicU8 = AtomicU8::new(0);
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static skyrt_cheat_god: AtomicU8 = AtomicU8::new(0);

static QUEUE: Mutex<Vec<Cheat>> = Mutex::new(Vec::new());
/// The run speed last asked for, in percent (for the hosts' menus).
static RUN_SPEED: AtomicU32 = AtomicU32::new(100);

/// Asks for `cheat`: done at the end of the frame at hand, or once a game is loaded.
pub fn request(cheat: Cheat) {
    log::info!("cheat requested: {cheat:?}");
    if let Cheat::RunSpeed(percent) = cheat {
        RUN_SPEED.store(percent as u32, Ordering::Relaxed);
    }
    QUEUE.lock().unwrap().push(cheat);
    skyrt_cheats_pending.store(1, Ordering::Release);
}

/// God mode: the player loses no health, magicka or stamina.
pub fn set_god_mode(on: bool) {
    log::info!("cheat: god mode {}", if on { "on" } else { "off" });
    skyrt_cheat_god.store(u8::from(on), Ordering::Relaxed);
}

pub fn god_mode() -> bool {
    skyrt_cheat_god.load(Ordering::Relaxed) != 0
}

pub fn run_speed() -> u32 {
    RUN_SPEED.load(Ordering::Relaxed)
}

/// A cheat by the name the hosts' command lines use: `god`, `god=off`, `gold=1000`,
/// `lockpicks=10`, `heal`, `carry=500`, `perks=1`, `skills=1`, `speed=150`, `status`.
pub fn request_named(text: &str) -> Result<(), String> {
    let (name, value) = text.split_once('=').map_or((text, None), |(n, v)| (n, Some(v)));
    let number = |default: f64| match value {
        Some(v) => v.parse::<f64>().map_err(|e| format!("cheat {text}: {e}")),
        None => Ok(default),
    };
    match name {
        "god" => set_god_mode(value != Some("off")),
        "gold" => request(Cheat::Gold(number(1000.0)? as u32)),
        "lockpicks" => request(Cheat::Lockpicks(number(10.0)? as u32)),
        "heal" => request(Cheat::Heal),
        "carry" => request(Cheat::CarryWeight(number(500.0)? as f32)),
        "perks" => request(Cheat::PerkPoints(number(1.0)? as u8)),
        "skills" => request(Cheat::SkillLevels(number(1.0)? as u32)),
        "speed" => request(Cheat::RunSpeed(number(150.0)? as f32)),
        "status" => request(Cheat::Status),
        _ => return Err(format!("unknown cheat {name:?}")),
    }
    Ok(())
}

/// Calls the title's function at `function` with `args` in r3 on (and `float` in f1); returns r3.
fn call(k: &Kernel, ctx: &mut PPCContext, function: u32, args: &[u32], float: Option<f32>) -> u32 {
    for (i, &a) in args.iter().enumerate() {
        ctx.set_gpr(3 + i, u64::from(a));
    }
    if let Some(value) = float {
        ctx.set_fpr_f64(1, f64::from(value));
    }
    if let Err(e) = k.rt.call(ctx, function) {
        log::error!("cheat: function {function:#010X}: {e}");
    }
    ctx.gpr32(3)
}

/// Calls virtual function `slot` of `object` with `args` after `object` itself.
fn call_virtual(k: &Kernel, ctx: &mut PPCContext, object: u32, slot: u32, args: &[u32], float: Option<f32>) -> u32 {
    let mem = k.mem();
    let function = mem.read_u32(mem.read_u32(object) + slot * 4);
    let mut all = vec![object];
    all.extend_from_slice(args);
    call(k, ctx, function, &all, float)
}

fn value(k: &Kernel, ctx: &mut PPCContext, player: u32, id: u32) -> f32 {
    call_virtual(k, ctx, player + VALUE_OWNER, GET_VALUE, &[id], None);
    ctx.fpr_f64(1) as f32
}

fn apply(k: &Kernel, ctx: &mut PPCContext, player: u32, cheat: Cheat) {
    let mem = k.mem();
    let owner = player + VALUE_OWNER;
    match cheat {
        Cheat::Gold(count) | Cheat::Lockpicks(count) => {
            let id = if matches!(cheat, Cheat::Gold(_)) { GOLD } else { LOCKPICK };
            let form = call(k, ctx, LOOKUP_FORM, &[id], None);
            if form == 0 {
                log::warn!("cheat: no form {id:#X}");
                return;
            }
            call(k, ctx, ADD_ITEM, &[player, form, count, 0, 0, 0], None);
        }
        Cheat::Heal => {
            for id in [HEALTH, MAGICKA, STAMINA] {
                call(k, ctx, RESTORE_VALUE, &[player, id], Some(1.0e6));
            }
        }
        Cheat::CarryWeight(amount) => {
            call_virtual(k, ctx, owner, MOD_VALUE, &[0, CARRY_WEIGHT], Some(amount));
        }
        Cheat::PerkPoints(count) => {
            let points = mem.read_u8(player + PERK_POINTS);
            mem.write_u8(player + PERK_POINTS, points.saturating_add(count));
        }
        Cheat::SkillLevels(levels) => {
            let skills = mem.read_u32(player + SKILLS);
            if skills == 0 {
                return;
            }
            for id in SKILL_VALUES {
                call(k, ctx, INCREMENT_SKILL, &[skills, id, levels], None);
            }
        }
        Cheat::RunSpeed(percent) => {
            call_virtual(k, ctx, owner, SET_VALUE, &[SPEED_MULT], Some(percent));
            // The game works the speed out again when the weight carried changes.
            for amount in [0.1, -0.1] {
                call_virtual(k, ctx, owner, MOD_VALUE, &[0, CARRY_WEIGHT], Some(amount));
            }
        }
        Cheat::Status => {}
    }
    log::info!(
        "cheat done: {cheat:?} (health {:.0}, magicka {:.0}, stamina {:.0}, carry weight {:.0}, speed {:.0}%, \
         {} gold, {} perk points, one-handed {:.0})",
        value(k, ctx, player, HEALTH),
        value(k, ctx, player, MAGICKA),
        value(k, ctx, player, STAMINA),
        value(k, ctx, player, CARRY_WEIGHT),
        value(k, ctx, player, SPEED_MULT),
        call(k, ctx, GOLD_AMOUNT, &[player], None),
        mem.read_u8(player + PERK_POINTS),
        value(k, ctx, player, SKILL_VALUES.start)
    );
}

/// Carries out the requested cheats; `ctx` is the calling thread's context, left as it was.
/// Called where the title ends a frame's job list, only when requests are waiting.
///
/// # Safety
/// `ctx` must point to the calling guest thread's live `PPCContext`, at a call boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn skyrt_cheats_tick(ctx: *mut PPCContext) {
    let Some(k) = Kernel::try_global() else { return };
    if crate::thread::current().is_none_or(|t| t.tid() != MAIN_THREAD) {
        return;
    }
    let mem = k.mem();
    let player = mem.read_u32(PLAYER);
    // Not in a game yet (the main menu, a load in progress): the requests wait.
    if player == 0 || mem.read_u32(player + PARENT_CELL) == 0 {
        return;
    }
    // Cleared first: a request made meanwhile raises it again.
    skyrt_cheats_pending.store(0, Ordering::Release);
    let cheats = std::mem::take(&mut *QUEUE.lock().unwrap());
    // SAFETY: the caller's.
    let ctx = unsafe { &mut *ctx };
    let saved = ctx.clone();
    for cheat in cheats {
        apply(k, ctx, player, cheat);
    }
    *ctx = saved;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_requests_are_queued_with_their_values() {
        request_named("gold=250").unwrap();
        request_named("heal").unwrap();
        request_named("speed").unwrap();
        assert!(request_named("fly").is_err());
        assert!(request_named("gold=lots").is_err());
        let queued = std::mem::take(&mut *QUEUE.lock().unwrap());
        assert_eq!(queued, [Cheat::Gold(250), Cheat::Heal, Cheat::RunSpeed(150.0)]);
        assert_eq!(run_speed(), 150);
        assert!(!god_mode());
        request_named("god").unwrap();
        assert!(god_mode());
        request_named("god=off").unwrap();
        assert!(!god_mode());
    }
}
