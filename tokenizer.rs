// BPE Tokenizer from scratch for lafontaine-gpt
// Compile: rustup run stable-x86_64-pc-windows-gnu rustc -O tokenizer.rs -o tokenizer_train.exe
// Run:     .\tokenizer_train.exe --vocab_size 32000 --min_freq 2

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use std::env;

const DEFAULT_VOCAB_SIZE: usize = 32000;
const DEFAULT_MIN_FREQ:   usize = 5;
const FABLES_DIR: &str = "Data - Fables";
const FRENCH_DIR: &str = "Data - French";
const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<unk>", "<bos>", "<eos>"];
const WORD_END: &str = "</w>";


// ── Corpus loading ─────────────────────────────────────────────────────────────

fn load_txt_files(dir: &str) -> String {
    let mut corpus = String::new();
    let mut count  = 0;
    if !Path::new(dir).exists() {
        eprintln!("  Warning: {} not found, skipping.", dir);
        return corpus;
    }
    visit_dirs(Path::new(dir), &mut |path| {
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            match fs::read_to_string(path) {
                Ok(content) => { corpus.push_str(&content); corpus.push('\n'); count += 1; }
                Err(e) => eprintln!("  Warning: could not read {:?}: {}", path, e),
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

fn get_word_freqs(text: &str, min_freq: usize) -> HashMap<Vec<String>, usize> {
    print!("  [1/4] Counting word frequencies... ");
    io::stdout().flush().unwrap();
    let t = Instant::now();

    let mut raw_freq: HashMap<String, usize> = HashMap::new();
    let mut current_word = String::new();
    let total_chars = text.len();
    let mut last_pct = 0usize;

    for (i, c) in text.chars().enumerate() {
        let lc = c.to_lowercase().next().unwrap_or(c);
        if is_french_char(lc) {
            current_word.push(lc);
        } else if !current_word.is_empty() {
            *raw_freq.entry(current_word.clone()).or_insert(0) += 1;
            current_word.clear();
        }
        let pct = i * 100 / total_chars;
        if pct >= last_pct + 10 {
            last_pct = pct;
            print!("{}%... ", pct);
            io::stdout().flush().unwrap();
        }
    }
    if !current_word.is_empty() {
        *raw_freq.entry(current_word).or_insert(0) += 1;
    }
    println!("done in {:.1}s ({} unique words)", t.elapsed().as_secs_f64(), raw_freq.len());

    print!("  [2/4] Building character-level vocab (min_freq={})... ", min_freq);
    io::stdout().flush().unwrap();
    let t = Instant::now();
    let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
    for (word, freq) in raw_freq {
        if freq >= min_freq {
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            symbols.push(WORD_END.to_string());
            *word_freqs.entry(symbols).or_insert(0) += freq;
        }
    }
    println!("done in {:.1}s ({} words kept)", t.elapsed().as_secs_f64(), word_freqs.len());
    word_freqs
}

fn get_pairs(word_freqs: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
    let mut pairs: HashMap<(String, String), usize> = HashMap::new();
    for (symbols, freq) in word_freqs {
        for i in 0..symbols.len().saturating_sub(1) {
            *pairs.entry((symbols[i].clone(), symbols[i+1].clone())).or_insert(0) += freq;
        }
    }
    pairs
}

fn merge_pair(pair: &(String, String), word_freqs: HashMap<Vec<String>, usize>) -> HashMap<Vec<String>, usize> {
    let merged = format!("{}{}", pair.0, pair.1);
    let mut new_word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
    for (symbols, freq) in word_freqs {
        let mut new_symbols: Vec<String> = Vec::with_capacity(symbols.len());
        let mut i = 0;
        while i < symbols.len() {
            if i + 1 < symbols.len() && symbols[i] == pair.0 && symbols[i+1] == pair.1 {
                new_symbols.push(merged.clone());
                i += 2;
            } else {
                new_symbols.push(symbols[i].clone());
                i += 1;
            }
        }
        *new_word_freqs.entry(new_symbols).or_insert(0) += freq;
    }
    new_word_freqs
}


// ── BPE Training ──────────────────────────────────────────────────────────────

fn train_bpe(text: &str, vocab_size: usize, min_freq: usize) -> (Vec<String>, Vec<(String, String)>) {
    println!("\nBPE training ===> target: {} tokens, min_freq: {}", vocab_size, min_freq);

    let mut word_freqs = get_word_freqs(text, min_freq);

    print!("  [3/4] Building base vocabulary... ");
    io::stdout().flush().unwrap();
    let mut base_chars: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for symbols in word_freqs.keys() {
        for s in symbols { base_chars.insert(s.clone()); }
    }
    let mut vocab_tokens: Vec<String> = SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect();
    vocab_tokens.extend(base_chars.into_iter());
    let n_merges = vocab_size.saturating_sub(vocab_tokens.len());
    println!("{} base tokens ===> {} merges to learn", vocab_tokens.len(), n_merges);

    println!("  [4/4] Learning merges...\n");
    io::stdout().flush().unwrap();

    let mut merges: Vec<(String, String)> = Vec::with_capacity(n_merges);
    let t0 = Instant::now();

    for i in 0..n_merges {
        let pairs = get_pairs(&word_freqs);
        if pairs.is_empty() { println!("No more pairs after {} merges.", i); break; }

        let best = pairs.into_iter().max_by_key(|(_, freq)| *freq).unwrap().0;
        word_freqs = merge_pair(&best, word_freqs);
        vocab_tokens.push(format!("{}{}", best.0, best.1));
        merges.push(best.clone());

        if (i + 1) % 500 == 0 {
            let elapsed   = t0.elapsed().as_secs_f64();
            let per_merge = elapsed / (i + 1) as f64;
            let remaining = per_merge * (n_merges - i - 1) as f64;
            let mins      = (remaining / 60.0) as u64;
            let secs      = (remaining % 60.0) as u64;
            println!(
                "  Merge {:6}/{} ===> {}m{}s remaining ===> \"{}\"",
                i + 1, n_merges, mins, secs,
                format!("{}{}", best.0, best.1)
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
    print!("Saving tokenizer ===> {}... ", path);
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