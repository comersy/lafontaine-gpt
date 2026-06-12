// BPE Tokenizer from scratch for lafontaine-gpt
// Parallelized encoding with rayon + dashmap
//
// Commands:
//   train  : learns BPE merges from corpus, outputs tokenizer.json
//   encode : encodes a corpus to binary ids using tokenizer.json
//
// Build:  cargo build --release
// Run:    .\target\release\tokenizer_train.exe train  --vocab_size 32000 --min_freq 2
//         .\target\release\tokenizer_train.exe encode --mode pretrain --output pretrain_ids.bin
//         .\target\release\tokenizer_train.exe encode --mode finetune --output finetune_ids.bin

use std::collections::{HashMap, BTreeSet};
use std::fs;
use std::io::{self, Write, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use rayon::prelude::*;
use dashmap::DashMap;

const DEFAULT_VOCAB_SIZE: usize = 32000;
const DEFAULT_MIN_FREQ:   usize = 2;
const FABLES_DIR: &str = "Data - Fables";
const FRENCH_DIR: &str = "Data - French";
const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<unk>", "<bos>", "<eos>"];
const WORD_END: &str = "</w>";
const UNK_ID: u16 = 1;
const BOS_ID: u16 = 2;
const EOS_ID: u16 = 3;


// ── Corpus loading ─────────────────────────────────────────────────────────────

fn collect_txt_files(dir: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !Path::new(dir).exists() {
        eprintln!("  Warning: {} not found", dir);
        return files;
    }
    visit_dirs(Path::new(dir), &mut |path| {
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            files.push(path.to_path_buf());
        }
    });
    files.sort();
    files
}

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


// ── Char helpers ───────────────────────────────────────────────────────────────

fn is_french_char(c: char) -> bool {
    c.is_ascii_alphabetic()
        || matches!(c,
            'À'|'Â'|'Ä'|'È'|'É'|'Ê'|'Ë'|'Î'|'Ï'|'Ô'|'Ù'|'Û'|'Ü'|'Ÿ'|'Ç'|'Œ'|'Æ'|
            'à'|'â'|'ä'|'è'|'é'|'ê'|'ë'|'î'|'ï'|'ô'|'ù'|'û'|'ü'|'ÿ'|'ç'|'œ'|'æ'|
            '\''|'\u{2019}'|'-'
        )
}


// ── Symbol vocab ──────────────────────────────────────────────────────────────

type SymId = u32;

struct SymVocab {
    sym_to_id : HashMap<String, SymId>,
    id_to_sym : Vec<String>,
}

impl SymVocab {
    fn new() -> Self { Self { sym_to_id: HashMap::new(), id_to_sym: Vec::new() } }
    fn get_or_insert(&mut self, s: &str) -> SymId {
        if let Some(&id) = self.sym_to_id.get(s) { return id; }
        let id = self.id_to_sym.len() as SymId;
        self.id_to_sym.push(s.to_string());
        self.sym_to_id.insert(s.to_string(), id);
        id
    }
    fn sym(&self, id: SymId) -> &str { &self.id_to_sym[id as usize] }
}


// ── TRAIN ─────────────────────────────────────────────────────────────────────

fn get_word_freqs(text: &str, min_freq: usize, vocab: &mut SymVocab) -> Vec<(Vec<SymId>, usize)> {
    print!("  [1/4] Counting word frequencies (parallel)... ");
    io::stdout().flush().unwrap();
    let t = Instant::now();

    let n_threads  = rayon::current_num_threads();
    let chunk_size = (text.len() / n_threads).max(1);

    // Split text into chunks at whitespace boundaries
    let mut chunks: Vec<&str> = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());
        while end < text.len() && !text.as_bytes()[end].is_ascii_whitespace() {
            end += 1;
        }
        chunks.push(&text[start..end]);
        start = end;
    }

    // Each thread counts its own chunk
    let partial_freqs: Vec<HashMap<String, usize>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut freq: HashMap<String, usize> = HashMap::new();
            let mut cur = String::new();
            for c in chunk.chars() {
                let lc = c.to_lowercase().next().unwrap_or(c);
                if is_french_char(lc) {
                    cur.push(lc);
                } else if !cur.is_empty() {
                    *freq.entry(cur.clone()).or_insert(0) += 1;
                    cur.clear();
                }
            }
            if !cur.is_empty() {
                *freq.entry(cur).or_insert(0) += 1;
            }
            freq
        })
        .collect();

    // Merge all partial frequency maps
    let mut raw_freq: HashMap<String, usize> = HashMap::new();
    for partial in partial_freqs {
        for (word, freq) in partial {
            *raw_freq.entry(word).or_insert(0) += freq;
        }
    }

    println!("done in {:.1}s ({} unique words)", t.elapsed().as_secs_f64(), raw_freq.len());

    print!("  [2/4] Building char-level word list (min_freq={})... ", min_freq);
    io::stdout().flush().unwrap();
    let t = Instant::now();
    let word_end_id = vocab.get_or_insert(WORD_END);
    let mut words: Vec<(Vec<SymId>, usize)> = Vec::new();
    for (word, freq) in raw_freq {
        if freq >= min_freq {
            let mut syms: Vec<SymId> = word.chars().map(|c| vocab.get_or_insert(&c.to_string())).collect();
            syms.push(word_end_id);
            words.push((syms, freq));
        }
    }
    println!("done in {:.1}s ({} words kept)", t.elapsed().as_secs_f64(), words.len());
    words
}

fn build_pair_counts(words: &[(Vec<SymId>, usize)]) -> HashMap<(SymId, SymId), usize> {
    let mut pairs: HashMap<(SymId, SymId), usize> = HashMap::new();
    for (syms, freq) in words {
        for i in 0..syms.len().saturating_sub(1) {
            *pairs.entry((syms[i], syms[i+1])).or_insert(0) += freq;
        }
    }
    pairs
}

fn apply_merge(
    words: &mut Vec<(Vec<SymId>, usize)>,
    pair_counts: &mut HashMap<(SymId, SymId), usize>,
    best: (SymId, SymId),
    new_id: SymId,
) {
    for (syms, freq) in words.iter_mut() {
        let mut i = 0;
        while i < syms.len().saturating_sub(1) {
            if syms[i] == best.0 && syms[i+1] == best.1 {
                if i > 0 {
                    let cnt = pair_counts.entry((syms[i-1], syms[i])).or_insert(0);
                    *cnt = cnt.saturating_sub(*freq);
                }
                if i + 2 < syms.len() {
                    let cnt = pair_counts.entry((syms[i+1], syms[i+2])).or_insert(0);
                    *cnt = cnt.saturating_sub(*freq);
                }
                syms[i] = new_id;
                syms.remove(i + 1);
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
    pair_counts.remove(&best);
}

fn train(vocab_size: usize, min_freq: usize) {
    println!("Loading corpus...");
    let t = Instant::now();
    let mut corpus = load_txt_files(FABLES_DIR);
    corpus.push_str(&load_txt_files(FRENCH_DIR));
    println!("Total ===> {} characters in {:.1}s\n", corpus.len(), t.elapsed().as_secs_f64());

    println!("BPE training ===> target: {} tokens, min_freq: {}", vocab_size, min_freq);
    let mut sym_vocab = SymVocab::new();
    let mut words     = get_word_freqs(&corpus, min_freq, &mut sym_vocab);

    print!("  [3/4] Building base vocabulary... ");
    io::stdout().flush().unwrap();
    let mut base_chars: BTreeSet<String> = BTreeSet::new();
    for (syms, _) in &words {
        for &id in syms { base_chars.insert(sym_vocab.sym(id).to_string()); }
    }
    let mut vocab_tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect();
    vocab_tokens.extend(base_chars.into_iter());
    let n_merges = vocab_size.saturating_sub(vocab_tokens.len());
    println!("{} base tokens ===> {} merges to learn", vocab_tokens.len(), n_merges);

    print!("  [4/4] Building initial pair counts... ");
    io::stdout().flush().unwrap();
    let t = Instant::now();
    let mut pair_counts = build_pair_counts(&words);
    println!("done in {:.1}s ({} unique pairs)\n", t.elapsed().as_secs_f64(), pair_counts.len());

    println!("  Learning merges...\n");
    let mut merges: Vec<(String, String)> = Vec::with_capacity(n_merges);
    let t0 = Instant::now();

    for i in 0..n_merges {
        let best = match pair_counts.iter().max_by_key(|(_, &v)| v) {
            Some((&k, _)) => k,
            None => { println!("No more pairs after {} merges.", i); break; }
        };
        let merged = format!("{}{}", sym_vocab.sym(best.0), sym_vocab.sym(best.1));
        let new_id = sym_vocab.get_or_insert(&merged);
        merges.push((sym_vocab.sym(best.0).to_string(), sym_vocab.sym(best.1).to_string()));
        vocab_tokens.push(merged.clone());
        apply_merge(&mut words, &mut pair_counts, best, new_id);

        if (i + 1) % 500 == 0 {
            let elapsed   = t0.elapsed().as_secs_f64();
            let per_merge = elapsed / (i + 1) as f64;
            let remaining = per_merge * (n_merges - i - 1) as f64;
            println!(
                "  Merge {:6}/{} ===> {}m{}s remaining ===> \"{}\"",
                i + 1, n_merges,
                (remaining / 60.0) as u64,
                (remaining % 60.0) as u64,
                merged
            );
            io::stdout().flush().unwrap();
        }
    }

    println!("\nVocabulary ===> {} tokens", vocab_tokens.len());
    save_tokenizer(&vocab_tokens, &merges, vocab_size, min_freq, "tokenizer.json");
}


// ── ENCODE (parallelized by chunks within each file) ─────────────────────────

struct BPEEncoder {
    vocab  : HashMap<String, u16>,
    merges : Vec<(String, String)>,
    cache  : Arc<DashMap<String, Vec<u16>>>,
}

impl BPEEncoder {
    fn load(path: &str) -> Self {
        let content = fs::read_to_string(path).expect("Cannot read tokenizer.json");
        let mut vocab: HashMap<String, u16> = HashMap::new();
        let mut merges: Vec<(String, String)> = Vec::new();

        let vocab_start = content.find("\"vocab\"").unwrap();
        for line in content[vocab_start..].lines().skip(2) {
            let line = line.trim();
            if line == "}" { break; }
            if let Some(colon) = line.find(':') {
                let key = line[..colon].trim().trim_matches('"').to_string();
                let val_str = line[colon+1..].trim().trim_end_matches(',');
                if let Ok(val) = val_str.parse::<u16>() {
                    vocab.insert(key, val);
                }
            }
        }

        let merges_start = content.find("\"merges\"").unwrap();
        let merges_end   = content.find("\"vocab\"").unwrap();
        for line in content[merges_start..merges_end].lines().skip(2) {
            let line = line.trim().trim_end_matches(',');
            if line == "]" { break; }
            if line.starts_with('[') {
                let inner = line.trim_start_matches('[').trim_end_matches(']');
                let parts: Vec<&str> = inner.split(',').collect();
                if parts.len() == 2 {
                    merges.push((
                        parts[0].trim().trim_matches('"').to_string(),
                        parts[1].trim().trim_matches('"').to_string(),
                    ));
                }
            }
        }

        println!("Tokenizer loaded ===> {} tokens, {} merges", vocab.len(), merges.len());
        Self { vocab, merges, cache: Arc::new(DashMap::new()) }
    }

    fn tokenize_word(&self, word: &str) -> Vec<u16> {
        if let Some(ids) = self.cache.get(word) {
            return ids.clone();
        }
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        symbols.push(WORD_END.to_string());
        for (a, b) in &self.merges {
            let mut i = 0;
            let mut new_syms: Vec<String> = Vec::with_capacity(symbols.len());
            while i < symbols.len() {
                if i + 1 < symbols.len() && &symbols[i] == a && &symbols[i+1] == b {
                    new_syms.push(format!("{}{}", a, b));
                    i += 2;
                } else {
                    new_syms.push(symbols[i].clone());
                    i += 1;
                }
            }
            symbols = new_syms;
        }
        let ids: Vec<u16> = symbols.iter()
            .map(|s| *self.vocab.get(s).unwrap_or(&UNK_ID))
            .collect();
        self.cache.insert(word.to_string(), ids.clone());
        ids
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u16> {
        let mut ids: Vec<u16> = Vec::new();
        let mut cur_word = String::new();
        for c in chunk.chars() {
            let lc = c.to_lowercase().next().unwrap_or(c);
            if is_french_char(lc) {
                cur_word.push(lc);
            } else {
                if !cur_word.is_empty() {
                    ids.extend(self.tokenize_word(&cur_word));
                    cur_word.clear();
                }
                if !c.is_whitespace() {
                    ids.push(*self.vocab.get(&c.to_string()).unwrap_or(&UNK_ID));
                }
            }
        }
        if !cur_word.is_empty() {
            ids.extend(self.tokenize_word(&cur_word));
        }
        ids
    }
}

fn split_into_chunks(text: &str, n: usize) -> Vec<&str> {
    let chunk_size = (text.len() / n).max(1);
    let mut chunks = Vec::new();
    let mut start  = 0;
    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());
        // Extend to next whitespace to avoid cutting mid-word
        while end < text.len() && !text.as_bytes()[end].is_ascii_whitespace() {
            end += 1;
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

fn encode(mode: &str, output: &str) {
    let encoder   = Arc::new(BPEEncoder::load("tokenizer.json"));
    let n_threads = rayon::current_num_threads();

    let (dir, add_bos_eos) = match mode {
        "pretrain" => (FRENCH_DIR, false),
        "finetune" => (FABLES_DIR, true),
        _ => panic!("mode must be pretrain or finetune"),
    };

    let all_files = collect_txt_files(dir);
    let total_bytes: u64 = all_files.iter()
        .map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    println!("\nEncoding {} corpus ===> {} ({} files, {:.2} GB) on {} threads",
        mode, output, all_files.len(), total_bytes as f64 / 1e9, n_threads);

    let t0         = Instant::now();
    let bytes_done = Arc::new(AtomicU64::new(0));
    let print_every: u64 = 50_000_000; // every 50MB
    let next_print = Arc::new(AtomicU64::new(print_every));

    let out_file   = fs::File::create(output).expect("Cannot create output file");
    let mut writer = BufWriter::new(out_file);
    let mut total_tokens: u64 = 0;

    for path in &all_files {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => { eprintln!("Warning: {:?}: {}", path, e); continue; }
        };

        // Split into n_threads chunks at word boundaries
        let chunks = split_into_chunks(&text, n_threads);

        let bd = Arc::clone(&bytes_done);
        let np = Arc::clone(&next_print);

        // Encode chunks in parallel
        let chunk_ids: Vec<Vec<u16>> = chunks
            .par_iter()
            .map(|chunk| {
                let enc        = Arc::clone(&encoder);
                let chunk_bytes = chunk.len() as u64;
                let ids        = enc.encode_chunk(chunk);

                // Update global byte counter
                let total_done = bd.fetch_add(chunk_bytes, Ordering::Relaxed) + chunk_bytes;
                let threshold  = np.load(Ordering::Relaxed);
                if total_done >= threshold {
                    if np.compare_exchange(threshold, threshold + print_every,
                        Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                        let pct       = total_done * 100 / total_bytes.max(1);
                        let elapsed   = t0.elapsed().as_secs_f64();
                        let remaining = elapsed / total_done as f64
                            * total_bytes.saturating_sub(total_done) as f64;
                        eprintln!("  [{:3}%] {:.2}/{:.2} GB ===> {}m{}s remaining ===> {} words cached",
                            pct,
                            total_done as f64 / 1e9,
                            total_bytes as f64 / 1e9,
                            (remaining / 60.0) as u64,
                            (remaining % 60.0) as u64,
                            enc.cache.len(),
                        );
                    }
                }
                ids
            })
            .collect();

        if add_bos_eos {
            writer.write_all(&BOS_ID.to_le_bytes()).unwrap();
            total_tokens += 1;
        }
        for ids in &chunk_ids {
            for id in ids {
                writer.write_all(&id.to_le_bytes()).unwrap();
            }
            total_tokens += ids.len() as u64;
        }
        if add_bos_eos {
            writer.write_all(&EOS_ID.to_le_bytes()).unwrap();
            total_tokens += 1;
        }
    }

    println!("\nEncoding done ===> {}M tokens in {:.1}s ===> {}",
        total_tokens / 1_000_000, t0.elapsed().as_secs_f64(), output);
    println!("Word cache ===> {} unique words", encoder.cache.len());
}


// ── JSON save ─────────────────────────────────────────────────────────────────

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
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  train  --vocab_size 32000 --min_freq 2");
        eprintln!("  encode --mode pretrain --output pretrain_ids.bin");
        eprintln!("  encode --mode finetune --output finetune_ids.bin");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            let mut vocab_size = DEFAULT_VOCAB_SIZE;
            let mut min_freq   = DEFAULT_MIN_FREQ;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--vocab_size" => { vocab_size = args[i+1].parse().unwrap(); i += 2; }
                    "--min_freq"   => { min_freq   = args[i+1].parse().unwrap(); i += 2; }
                    _ => { i += 1; }
                }
            }
            train(vocab_size, min_freq);
        }
        "encode" => {
            let mut mode   = "pretrain".to_string();
            let mut output = "pretrain_ids.bin".to_string();
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--mode"   => { mode   = args[i+1].clone(); i += 2; }
                    "--output" => { output = args[i+1].clone(); i += 2; }
                    _ => { i += 1; }
                }
            }
            encode(&mode, &output);
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            std::process::exit(1);
        }
    }
}