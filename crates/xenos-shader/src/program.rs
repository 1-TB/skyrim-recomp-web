//! Whole-program decoding: control flow list, labels and the instructions of every exec.

use std::collections::BTreeSet;

use crate::ShaderStage;
use crate::parsed::{
    self, ParsedAlu, ParsedCf, ParsedTextureFetch, ParsedVertexFetch, parse_alu, parse_cf, parse_texture_fetch,
    parse_vertex_fetch,
};
use crate::ucode::{FetchOpcode, RawAlu, RawFetch, unpack_cf_pair};

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Alu(ParsedAlu),
    VertexFetch(ParsedVertexFetch),
    TextureFetch(ParsedTextureFetch),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecInstruction {
    /// Instruction slot (dword offset / 3).
    pub address: u32,
    pub serialize: bool,
    pub instruction: Instruction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CfEntry {
    /// Control flow instruction index (pair index * 2 + 0/1).
    pub index: u32,
    pub cf: ParsedCf,
    /// Instructions executed by an exec (empty for other control flow instructions).
    pub instructions: Vec<ExecInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UcodeError {
    Empty,
    /// An exec references instruction slots outside the microcode.
    InstructionOutOfRange {
        cf_index: u32,
        address: u32,
    },
}

impl std::fmt::Display for UcodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UcodeError::Empty => write!(f, "shader microcode is empty"),
            UcodeError::InstructionOutOfRange { cf_index, address } => {
                write!(f, "control flow instruction {cf_index} executes slot {address} outside the microcode")
            }
        }
    }
}

impl std::error::Error for UcodeError {}

#[derive(Clone, Debug)]
pub struct Program {
    pub stage: ShaderStage,
    pub ucode: Vec<u32>,
    /// Number of 3-dword control flow pairs before the first exec'd instruction.
    pub cf_pair_bound: u32,
    pub cf: Vec<CfEntry>,
    /// Targets of calls, jumps and loops (as printed by the disassembler).
    pub labels: BTreeSet<u32>,
}

fn instruction_dwords(ucode: &[u32], address: u32) -> Option<[u32; 3]> {
    let start = address as usize * 3;
    ucode.get(start..start + 3).map(|s| [s[0], s[1], s[2]])
}

impl Program {
    pub fn decode(stage: ShaderStage, ucode: &[u32]) -> Result<Program, UcodeError> {
        if ucode.len() < 3 {
            return Err(UcodeError::Empty);
        }
        let mut cf_pair_bound = (ucode.len() / 3) as u32;
        let mut labels = BTreeSet::new();
        let mut i = 0;
        while i < cf_pair_bound {
            let pair = unpack_cf_pair(instruction_dwords(ucode, i).unwrap());
            for cf in pair {
                if cf.opcode().is_exec() {
                    cf_pair_bound = cf_pair_bound.min(cf.exec_address());
                }
                match parse_cf(&cf, 0) {
                    ParsedCf::Call { target, .. } | ParsedCf::Jump { target, .. } => {
                        labels.insert(target);
                    }
                    ParsedCf::LoopStart { skip_address, .. } => {
                        labels.insert(skip_address);
                    }
                    ParsedCf::LoopEnd { body_address, .. } => {
                        labels.insert(body_address);
                    }
                    _ => {}
                }
            }
            i += 1;
        }

        let mut cf_list = Vec::with_capacity(cf_pair_bound as usize * 2);
        let mut previous_vfetch_full = RawFetch::default();
        for pair_index in 0..cf_pair_bound {
            let pair = unpack_cf_pair(instruction_dwords(ucode, pair_index).unwrap());
            for (j, raw) in pair.iter().enumerate() {
                let index = pair_index * 2 + j as u32;
                let cf = parse_cf(raw, index);
                let mut instructions = Vec::new();
                if let ParsedCf::Exec(exec) = &cf {
                    for k in 0..exec.count {
                        let address = exec.address + k;
                        let dwords = instruction_dwords(ucode, address)
                            .ok_or(UcodeError::InstructionOutOfRange { cf_index: index, address })?;
                        let instruction = if exec.is_fetch(k) {
                            let fetch = RawFetch(dwords);
                            if fetch.opcode() == FetchOpcode::VertexFetch {
                                let parsed = parse_vertex_fetch(&fetch, &previous_vfetch_full);
                                if !fetch.vf_is_mini_fetch() {
                                    previous_vfetch_full = fetch;
                                }
                                Instruction::VertexFetch(parsed)
                            } else {
                                Instruction::TextureFetch(parse_texture_fetch(&fetch))
                            }
                        } else {
                            Instruction::Alu(parse_alu(&RawAlu(dwords), stage))
                        };
                        instructions.push(ExecInstruction { address, serialize: exec.is_serialize(k), instruction });
                    }
                }
                cf_list.push(CfEntry { index, cf, instructions });
            }
        }
        Ok(Program { stage, ucode: ucode.to_vec(), cf_pair_bound, cf: cf_list, labels })
    }

    /// All instructions of all execs in control flow order.
    pub fn instructions(&self) -> impl Iterator<Item = (&CfEntry, &ExecInstruction)> {
        self.cf.iter().flat_map(|entry| entry.instructions.iter().map(move |i| (entry, i)))
    }

    pub fn cf_count(&self) -> u32 {
        self.cf.len() as u32
    }
}

pub use parsed::ParsedExec;
