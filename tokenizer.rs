// BPE Tokenizer from scratch for lafontaine-gpt
// Optimized: pair counts updated incrementally instead of recomputed from scratch
//
// Compile: rustup run stable-x86_64-pc-windows-gnu rustc -O tokenizer.rs -o tokenizer_train.exe
// Run:     .\tokenizer_train.exe --vocab_size 32000 --min_freq 2

use std::collections::{HashMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use std::env;

const DEFAULT_VOCAB_SIZE: usize = 32000;
const DEFAULT_MIN_FREQ:   usize = 2;
const FABLES_DIR: &str = "Data - Fables";
const FRENCH_DIR: &str = "Data - French";
const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<unk>", "<bos>", "<eos>"];
const WORD_END: &str = "</w>";


// ── Corpus loading ─────────────────────────────────────────────────────────────

fn load_txt_files(dir: &str) -> String {
    let mut corpus = String::new();
    let mut count  = 0;
    if !Path::new(dir).exists() {
        eprintln!("  Warning: {} not found", dir);
        return corpus;
    }
    visit_dirs(Path::new(dir), &mut |path| {
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            if let Ok(content) = fs::read_to_string(path) {
                corpus.push_str(&content);
                corpus.push('\n');
                count += 1;
            }
        }
    });
    println!("  {} ===> {} files, {} chars", dir, count, corpus.len());
    corpus
}

fn visit_dirs(dir: &Path, cb: &mut impl FnMut(&Path)) {
    if let Ok(entries) = fs::read_dir(dir) {
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() { visit_dirs(&path, cb); } else { cb(&path); }
        }
    }
}


// ── Helpers ────────────────────────────────────────────────────────────────────

fn is_french_char(c: char) -> bool {
    c.is_ascii_alphabetic()
        || matches!(c,
            'À'|'Â'|'Ä'|'È'|'É'|'Ê'|'Ë'|'Î'|'Ï'|'Ô'|'Ù'|'Û'|'Ü'|'Ÿ'|'Ç'|'Œ'|'Æ'|
            'à'|'â'|'ä'|'è'|'é'|'ê'|'ë'|'î'|'ï'|'ô'|'ù'|'û'|'ü'|'ÿ'|'ç'|'œ'|'æ'|
            '\''|'\u{2019}'|'-'
        )
}

// Word represented as a Vec of symbol ids (faster than strings)
type SymId = u32;
type Word  = Vec<SymId>;

struct Vocab {
    sym_to_id : HashMap<String, SymId>,
    id_to_sym : Vec<String>,
}

impl Vocab {
    fn new() -> Self {
        Self { sym_to_id: HashMap::new(), id_to_sym: Vec::new() }
    }
    fn get_or_insert(&mut self, s: &str) -> SymId {
        if let Some(&id) = self.sym_to_id.get(s) {
            return id;
        }
        let id = self.id_to_sym.len() as SymId;
        self.id_to_sym.push(s.to_string());
        self.sym_to_id.insert(s.to_string(), id);
        id
    }
    fn sym(&self, id: SymId) -> &str {
        &self.id_to_sym[id as usize]
    }
}


// ── Step 1: build word frequencies ───────────────────────────────────────────

fn get_word_freqs(text: &str, min_freq: usize, vocab: &mut Vocab) -> Vec<(Word, usize)> {
    print!("  [1/4] Counting word frequencies... ");
    io::stdout().flush().unwrap();
    let t = Instant::now();

    let mut raw_freq: HashMap<String, usize> = HashMap::new();
    let mut cur = String::new();
    let total = text.len();
    let mut last_pct = 0usize;

    for (i, c) in text.chars().enumerate() {
        let lc = c.to_lowercase().next().unwrap_or(c);
        if is_french_char(lc) {
            cur.push(lc);
        } else if !cur.is_empty() {
            *raw_freq.entry(cur.clone()).or_insert(0) += 1;
            cur.clear();
        }
        let pct = i * 100 / total;
        if pct >= last_pct + 10 { last_pct = pct; print!("{}%... ", pct); io::stdout().flush().unwrap(); }
    }
    if !cur.is_empty() { *raw_freq.entry(cur).or_insert(0) += 1; }
    println!("done in {:.1}s ({} unique words)", t.elapsed().as_secs_f64(), raw_freq.len());

    print!("  [2/4] Building char-level word list (min_freq={})... ", min_freq);
    io::stdout().flush().unwrap();
    let t = Instant::now();

    let word_end_id = vocab.get_or_insert(WORD_END);

    let mut words: Vec<(Word, usize)> = Vec::new();
    for (word, freq) in raw_freq {
        if freq >= min_freq {
            let mut syms: Word = word.chars().map(|c| vocab.get_or_insert(&c.to_string())).collect();
            syms.push(word_end_id);
            words.push((syms, freq));
        }
    }
    println!("done in {:.1}s ({} words kept)", t.elapsed().as_secs_f64(), words.len());
    words
}


// ── Step 2: build initial pair counts ────────────────────────────────────────

fn build_pair_counts(words: &[(Word, usize)]) -> HashMap<(SymId, SymId), usize> {
    let mut pairs: HashMap<(SymId, SymId), usize> = HashMap::new();
    for (syms, freq) in words {
        for i in 0..syms.len().saturating_sub(1) {
            *pairs.entry((syms[i], syms[i+1])).or_insert(0) += freq;
        }
    }
    pairs
}


// ── Step 3: incremental merge ─────────────────────────────────────────────────
// Instead of recomputing all pairs, we only update pairs affected by the merge.

fn apply_merge(
    words       : &mut Vec<(Word, usize)>,
    pair_counts : &mut HashMap<(SymId, SymId), usize>,
    best        : (SymId, SymId),
    new_id      : SymId,
) {
    for (syms, freq) in words.iter_mut() {
        let mut i = 0;
        while i < syms.len().saturating_sub(1) {
            if syms[i] == best.0 && syms[i+1] == best.1 {
                // Remove old pairs around the merge site
                if i > 0 {
                    let cnt = pair_counts.entry((syms[i-1], syms[i])).or_insert(0);
                    *cnt = cnt.saturating_sub(*freq);
                }
                if i + 2 < syms.len() {
                    let cnt = pair_counts.entry((syms[i+1], syms[i+2])).or_insert(0);
                    *cnt = cnt.saturating_sub(*freq);
                }

                // Merge
                syms[i] = new_id;
                syms.remove(i + 1);

                // Add new pairs around merge site
                if i > 0 {
                    *pair_counts.entry((syms[i-1], new_id)).or_insert(0) += *freq;
                }
                if i + 1 < syms.len() {
                    *pair_counts.entry((new_id, syms[i+1])).or_insert(0) += *freq;
                }
            } else {
                i += 1;
            }
        }
    }
    // Remove the merged pair itself
    pair_counts.remove(&best);
}


// ── BPE Training ──────────────────────────────────────────────────────────────

fn train_bpe(text: &str, vocab_size: usize, min_freq: usize) -> (Vec<String>, Vec<(String, String)>) {
    println!("\nBPE training ===> target: {} tokens, min_freq: {}", vocab_size, min_freq);

    let mut sym_vocab = Vocab::new();
    let mut words     = get_word_freqs(text, min_freq, &mut sym_vocab);

    // Base vocab: special tokens + all chars
    print!("  [3/4] Building base vocabulary... ");
    io::stdout().flush().unwrap();

    let mut base_chars: BTreeSet<String> = BTreeSet::new();
    for (syms, _) in &words {
        for &id in syms {
            base_chars.insert(sym_vocab.sym(id).to_string());
        }
    }
    let mut vocab_tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect();
    vocab_tokens.extend(base_chars.into_iter());
    let n_merges = vocab_size.saturating_sub(vocab_tokens.len());
    println!("{} base tokens ===> {} merges to learn", vocab_tokens.len(), n_merges);

    // Build initial pair counts ONCE
    print!("  [4/4] Building initial pair counts... ");
    io::stdout().flush().unwrap();
    let t = Instant::now();
    let mut pair_counts = build_pair_counts(&words);
    println!("done in {:.1}s ({} unique pairs)", t.elapsed().as_secs_f64(), pair_counts.len());

    println!("\n  Learning merges...\n");
    io::stdout().flush().unwrap();

    let mut merges: Vec<(String, String)> = Vec::with_capacity(n_merges);
    let t0 = Instant::now();

    for i in 0..n_merges {
        // Find best pair
        let best = match pair_counts.iter().max_by_key(|(_, &v)| v) {
            Some((&k, _)) => k,
            None => { println!("No more pairs after {} merges.", i); break; }
        };

        let merged = format!("{}{}", sym_vocab.sym(best.0), sym_vocab.sym(best.1));
        let new_id = sym_vocab.get_or_insert(&merged);

        merges.push((sym_vocab.sym(best.0).to_string(), sym_vocab.sym(best.1).to_string()));
        vocab_tokens.push(merged.clone());

        // Incremental update — no full rescan
        apply_merge(&mut words, &mut pair_counts, best, new_id);

        if (i + 1) % 500 == 0 {
            let elapsed   = t0.elapsed().as_secs_f64();
            let per_merge = elapsed / (i + 1) as f64;
            let remaining = per_merge * (n_merges - i - 1) as f64;
            let mins      = (remaining / 60.0) as u64;
            let secs      = (remaining % 60.0) as u64;
            println!(
                "  Merge {:6}/{} ===> {}m{}s remaining ===> \"{}\"",
                i + 1, n_merges, mins, secs, merged
            );
            io::stdout().flush().unwrap();
        }
    }

    println!("\nVocabulary ===> {} tokens", vocab_tokens.len());
    (vocab_tokens, merges)
}


// ── JSON output ───────────────────────────────────────────────────────────────

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c    => out.push(c),
        }
    }
    out.push('"');
    out
}

fn save_tokenizer(vocab_tokens: &[String], merges: &[(String, String)], vocab_size: usize, min_freq: usize, path: &str) {
    print!("\nSaving tokenizer ===> {}... ", path);
    io::stdout().flush().unwrap();
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"vocab_size\": {},\n", vocab_size));
    json.push_str(&format!("  \"min_freq\": {},\n", min_freq));
    json.push_str("  \"merges\": [\n");
    for (i, (a, b)) in merges.iter().enumerate() {
        let comma = if i < merges.len() - 1 { "," } else { "" };
        json.push_str(&format!("    [{}, {}]{}\n", escape_json(a), escape_json(b), comma));
    }
    json.push_str("  ],\n");
    json.push_str("  \"vocab\": {\n");
    for (i, tok) in vocab_tokens.iter().enumerate() {
        let comma = if i < vocab_tokens.len() - 1 { "," } else { "" };
        json.push_str(&format!("    {}: {}{}\n", escape_json(tok), i, comma));
    }
    json.push_str("  }\n}\n");
    fs::write(path, json).expect("Could not write tokenizer.json");
    println!("done ({} tokens)", vocab_tokens.len());
}


// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut vocab_size = DEFAULT_VOCAB_SIZE;
    let mut min_freq   = DEFAULT_MIN_FREQ;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab_size" => { vocab_size = args[i+1].parse().unwrap(); i += 2; }
            "--min_freq"   => { min_freq   = args[i+1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("Loading corpus...");
    let t = Instant::now();
    let mut corpus = load_txt_files(FABLES_DIR);
    corpus.push_str(&load_txt_files(FRENCH_DIR));
    println!("Total ===> {} characters in {:.1}s\n", corpus.len(), t.elapsed().as_secs_f64());

    let (vocab_tokens, merges) = train_bpe(&corpus, vocab_size, min_freq);
    save_tokenizer(&vocab_tokens, &merges, vocab_size, min_freq, "tokenizer.json");
}