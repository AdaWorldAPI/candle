//! Qualia XOR: Compare Nib4 interior-physics encoding against
//! BERT/MiniLM text embeddings to measure structural truth.
//!
//! The XOR surfaces three buckets:
//!   A) Agree:              close in both → robust truth
//!   B) Nib4 close, BERT far: cadence truth language misses
//!   C) BERT close, Nib4 far: surface synonymy masking different physics

use candle_transformers::models::bert::{BertModel, Config, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::collections::HashMap;
use tokenizers::{PaddingParams, Tokenizer};

// ============================================================================
// NIB4 codebook (same as bf16_test)
// ============================================================================

const NIB4_LEVELS: u8 = 15;
const QUALIA_DIMS: usize = 16;

struct Nib4Codebook {
    bounds: Vec<(f32, f32)>,
}

impl Nib4Codebook {
    fn from_corpus(vectors: &[Vec<f32>]) -> Self {
        let ndims = vectors[0].len();
        let mut bounds = Vec::with_capacity(ndims);
        for d in 0..ndims {
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for v in vectors {
                if v[d] < mn { mn = v[d]; }
                if v[d] > mx { mx = v[d]; }
            }
            if (mx - mn).abs() < 1e-9 { mx = mn + 1.0; }
            bounds.push((mn, mx));
        }
        Self { bounds }
    }

    fn encode_dim(&self, dim: usize, val: f32) -> u8 {
        let (mn, mx) = self.bounds[dim];
        let t = (val - mn) / (mx - mn);
        (t * NIB4_LEVELS as f32).round().clamp(0.0, NIB4_LEVELS as f32) as u8
    }

    fn encode_vec(&self, vals: &[f32]) -> Vec<u8> {
        vals.iter().enumerate().map(|(d, &v)| self.encode_dim(d, v)).collect()
    }
}

fn nib4_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.abs_diff(y) as u32).sum()
}

// ============================================================================
// Corpus parsing
// ============================================================================

#[derive(Deserialize)]
struct QualiaItem {
    id: String,
    label: String,
    family: String,
    #[serde(default)]
    subfamily: Option<String>,
    vector: HashMap<String, f64>,
    #[serde(default)]
    qualia: Option<Vec<String>>,
    #[serde(default)]
    melodic_motions: Option<Vec<String>>,
    #[serde(default)]
    harmonic_bias: Option<Vec<String>>,
    #[serde(default)]
    truth_anchor_candidates: Option<Vec<String>>,
    #[serde(default)]
    neighbors: Option<Vec<String>>,
}

const DIMS_16_JSON: &[&str] = &[
    "brightness", "valence", "dominance", "arousal",
    "warmth", "clarity", "social", "nostalgia",
    "sacredness", "desire", "tension", "awe",
    "grief", "hope", "edge", "resolution_hunger",
];

const DIMS_16_NAMES: &[&str] = &[
    "glow", "valence", "rooting", "agency",
    "resonance", "clarity", "social", "gravity",
    "reverence", "volition", "dissonance", "staunen",
    "loss", "optimism", "friction", "equilibrium",
];

const DIM_INTENSITY: &str = "shame";

fn extract_16(item: &QualiaItem) -> Vec<f32> {
    DIMS_16_JSON.iter().map(|d| *item.vector.get(*d).unwrap_or(&0.0) as f32).collect()
}

fn extract_intensity_val(item: &QualiaItem) -> f32 {
    *item.vector.get(DIM_INTENSITY).unwrap_or(&0.0) as f32
}

/// Build rich text for embedding: family + label + qualia tags
/// This gives BERT much more semantic signal than just the item ID.
fn build_embedding_text(item: &QualiaItem) -> String {
    let mut parts = vec![
        format!("family: {}", item.family),
        format!("item: {}", item.label),
    ];

    if let Some(ref qualia) = item.qualia {
        parts.push(format!("qualia: {}", qualia.join(", ")));
    }

    if let Some(ref motions) = item.melodic_motions {
        parts.push(format!("motion: {}", motions.join(", ")));
    }

    if let Some(ref bias) = item.harmonic_bias {
        parts.push(format!("harmonic: {}", bias.join(", ")));
    }

    parts.join(" | ")
}

// ============================================================================
// BERT embedding via candle
// ============================================================================

fn load_bert_model(model_dir: &str) -> Result<(BertModel, Tokenizer)> {
    let device = candle_core::Device::Cpu;
    let model_dir = std::path::Path::new(model_dir);

    println!("  Loading config...");
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;

    println!("  Loading tokenizer...");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    println!("  Loading model weights ({}MB)...",
        std::fs::metadata(model_dir.join("model.safetensors"))?.len() / 1_000_000);
    let weights_path = model_dir.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

/// Embed a batch of sentences, return [N, hidden_size] tensor
fn embed_sentences(
    model: &BertModel,
    tokenizer: &mut Tokenizer,
    sentences: &[String],
) -> Result<Vec<Vec<f32>>> {
    let device = &model.device;

    // Set up padding for batch processing
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    // Process in batches to avoid OOM
    let batch_size = 32;
    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(sentences.len());

    for chunk_start in (0..sentences.len()).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(sentences.len());
        let chunk: Vec<String> = sentences[chunk_start..chunk_end].to_vec();

        let tokens = tokenizer
            .encode_batch(chunk.clone(), true)
            .map_err(|e| E::msg(e.to_string()))?;

        let token_ids = tokens
            .iter()
            .map(|t| {
                let ids = t.get_ids().to_vec();
                Ok(Tensor::new(ids.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|t| {
                let mask = t.get_attention_mask().to_vec();
                Ok(Tensor::new(mask.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // Forward pass
        let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling with attention mask
        let attention_mask_f = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let sum_mask = attention_mask_f.sum(1)?;
        let pooled = (embeddings.broadcast_mul(&attention_mask_f)?).sum(1)?;
        let pooled = pooled.broadcast_div(&sum_mask)?;

        // L2 normalize
        let pooled = normalize_l2(&pooled)?;

        // Extract to Vec<Vec<f32>>
        let pooled_data = pooled.to_vec2::<f32>()?;
        all_embeddings.extend(pooled_data);

        if chunk_end < sentences.len() {
            print!("  Embedded {}/{}\r", chunk_end, sentences.len());
        }
    }
    println!("  Embedded {}/{} sentences", all_embeddings.len(), sentences.len());

    Ok(all_embeddings)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

// ============================================================================
// Distance and rank comparison
// ============================================================================

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 { return 1.0; }
    1.0 - dot / (na * nb)
}

/// Kendall tau rank correlation between two orderings
fn kendall_tau(rank_a: &[usize], rank_b: &[usize]) -> f64 {
    let n = rank_a.len();
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let a_cmp = (rank_a[i] as i64 - rank_a[j] as i64).signum();
            let b_cmp = (rank_b[i] as i64 - rank_b[j] as i64).signum();
            if a_cmp == b_cmp {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let total = concordant + discordant;
    if total == 0 { return 0.0; }
    (concordant - discordant) as f64 / total as f64
}

/// Spearman footrule: sum of |rank_a[i] - rank_b[i]|
fn spearman_footrule(rank_a: &[usize], rank_b: &[usize]) -> f64 {
    let n = rank_a.len();
    let sum: usize = rank_a.iter().zip(rank_b).map(|(&a, &b)| {
        if a > b { a - b } else { b - a }
    }).sum();
    // Normalize by max possible (n²/2 for even n)
    let max_footrule = if n % 2 == 0 { n * n / 2 } else { (n * n - 1) / 2 };
    sum as f64 / max_footrule as f64
}

/// Get rank ordering for row i of a distance matrix
fn get_rank_order(distances: &[Vec<f32>], i: usize) -> Vec<usize> {
    let n = distances[i].len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| distances[i][a].partial_cmp(&distances[i][b]).unwrap());

    // Convert to ranks
    let mut ranks = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank;
    }
    ranks
}

fn main() -> Result<()> {
    // ========================================================================
    // LOAD CORPUS
    // ========================================================================
    let json_str = include_str!("qualia_219.json");
    let items: Vec<QualiaItem> = serde_json::from_str(json_str)?;
    let n = items.len();
    println!("=== QUALIA XOR: Nib4 vs BERT Truth Comparison ===");
    println!("  Corpus: {} items\n", n);

    // ========================================================================
    // STEP 1: Nib4 encoding (interior physics)
    // ========================================================================
    println!("--- Step 1: Nib4 encoding (interior physics) ---\n");

    let vecs_16: Vec<Vec<f32>> = items.iter().map(|it| extract_16(it)).collect();
    let intensity_vals: Vec<f32> = items.iter().map(|it| extract_intensity_val(it)).collect();

    let codebook = Nib4Codebook::from_corpus(&vecs_16);
    let nib4_vecs: Vec<Vec<u8>> = vecs_16.iter().map(|v| codebook.encode_vec(v)).collect();

    // Compute intensity bit: nonzero shame = CMYK (caused/absorbing)
    // Median is 0.0 (130 items have zero shame), so use strict > 0.0
    let intensity_bits: Vec<bool> = intensity_vals.iter().map(|&v| v > 0.0).collect();
    let n_cmyk = intensity_bits.iter().filter(|&&b| b).count();
    let n_rgb = n - n_cmyk;
    println!("  Mode split: {} RGB ({:.0}%) / {} CMYK ({:.0}%)",
        n_rgb, 100.0 * n_rgb as f64 / n as f64,
        n_cmyk, 100.0 * n_cmyk as f64 / n as f64);

    // Compute Nib4 pairwise distance matrix
    let mut nib4_dist: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = nib4_distance(&nib4_vecs[i], &nib4_vecs[j]) as f32;
            // Add intensity penalty if causality direction flips
            let penalty = if intensity_bits[i] != intensity_bits[j] { 16.0 } else { 0.0 };
            let total = d + penalty;
            nib4_dist[i][j] = total;
            nib4_dist[j][i] = total;
        }
    }
    println!("  Nib4 distance matrix: {}x{}", n, n);

    // ========================================================================
    // STEP 2: BERT embedding (observer language)
    // ========================================================================
    println!("\n--- Step 2: BERT embedding (observer language) ---\n");

    // Build rich text descriptions for each item
    let texts: Vec<String> = items.iter().map(|it| build_embedding_text(it)).collect();

    // Show a few examples
    println!("  Example embedding texts:");
    for i in [0, 50, 100, 150, 200].iter() {
        if *i < n {
            println!("    [{}] {}", i, &texts[*i][..texts[*i].len().min(80)]);
        }
    }
    println!();

    // Load model from local files
    let model_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/model");
    println!("  Loading model: all-MiniLM-L6-v2 (local)");
    let (model, mut tokenizer) = load_bert_model(model_dir)?;

    println!("  Embedding {} sentences...", n);
    let bert_embeddings = embed_sentences(&model, &mut tokenizer, &texts)?;
    let embed_dim = bert_embeddings[0].len();
    println!("  BERT embedding dimension: {}", embed_dim);

    // Compute BERT pairwise distance matrix (cosine distance)
    let mut bert_dist: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = cosine_distance(&bert_embeddings[i], &bert_embeddings[j]);
            bert_dist[i][j] = d;
            bert_dist[j][i] = d;
        }
    }
    println!("  BERT distance matrix: {}x{}", n, n);

    // ========================================================================
    // STEP 3: Rank-order comparison
    // ========================================================================
    println!("\n--- Step 3: Rank-order comparison ---\n");

    // Compute Kendall tau for each item
    let mut taus: Vec<(usize, f64)> = Vec::with_capacity(n);
    let mut footrules: Vec<(usize, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        let rank_nib = get_rank_order(&nib4_dist, i);
        let rank_bert = get_rank_order(&bert_dist, i);
        let tau = kendall_tau(&rank_nib, &rank_bert);
        let foot = spearman_footrule(&rank_nib, &rank_bert);
        taus.push((i, tau));
        footrules.push((i, foot));
    }

    // Distribution statistics
    let tau_values: Vec<f64> = taus.iter().map(|&(_, t)| t).collect();
    let mean_tau: f64 = tau_values.iter().sum::<f64>() / n as f64;
    let mut sorted_taus = tau_values.clone();
    sorted_taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_tau = sorted_taus[n / 2];
    let min_tau = sorted_taus[0];
    let max_tau = sorted_taus[n - 1];

    println!("  Kendall tau distribution (Nib4 vs BERT rank agreement):");
    println!("    mean:   {:.4}", mean_tau);
    println!("    median: {:.4}", median_tau);
    println!("    min:    {:.4}", min_tau);
    println!("    max:    {:.4}", max_tau);

    // Histogram
    println!("\n  Tau histogram:");
    let bins = [-1.0, -0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0];
    for w in bins.windows(2) {
        let count = tau_values.iter().filter(|&&t| t >= w[0] && t < w[1]).count();
        let bar = "#".repeat(count);
        println!("    [{:>+5.2}, {:>+5.2}): {:>3} {}", w[0], w[1], count, bar);
    }

    // Most stable items (highest tau = best agreement)
    let mut taus_sorted = taus.clone();
    taus_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  TOP 15 most stable items (highest tau = Nib4 and BERT agree):");
    println!("    {:<45}  {:>7}  {:>12}  {:>4}", "Item", "Tau", "Family", "Mode");
    for &(idx, tau) in taus_sorted.iter().take(15) {
        let mode = if intensity_bits[idx] { "CMYK" } else { "RGB" };
        println!("    {:<45}  {:>+7.4}  {:>12}  {:>4}",
            items[idx].id, tau, items[idx].family, mode);
    }

    println!("\n  BOTTOM 15 most unstable items (lowest tau = max disagreement):");
    println!("    {:<45}  {:>7}  {:>12}  {:>4}", "Item", "Tau", "Family", "Mode");
    for &(idx, tau) in taus_sorted.iter().rev().take(15) {
        let mode = if intensity_bits[idx] { "CMYK" } else { "RGB" };
        println!("    {:<45}  {:>+7.4}  {:>12}  {:>4}",
            items[idx].id, tau, items[idx].family, mode);
    }

    // ========================================================================
    // STEP 4: Three XOR buckets
    // ========================================================================
    println!("\n--- Step 4: XOR buckets (the interesting part) ---\n");

    let k = 10; // top-k neighborhood

    // For each pair, check if it's in top-k of one system but not the other
    let mut bucket_a: Vec<(usize, usize, f32, f32)> = Vec::new(); // agree
    let mut bucket_b: Vec<(usize, usize, f32, f32)> = Vec::new(); // nib4 close, bert far
    let mut bucket_c: Vec<(usize, usize, f32, f32)> = Vec::new(); // bert close, nib4 far

    for i in 0..n {
        // Get top-k neighbors in each system
        let mut nib4_neighbors: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, nib4_dist[i][j]))
            .collect();
        nib4_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nib4_topk: Vec<usize> = nib4_neighbors.iter().take(k).map(|&(j, _)| j).collect();

        let mut bert_neighbors: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, bert_dist[i][j]))
            .collect();
        bert_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bert_topk: Vec<usize> = bert_neighbors.iter().take(k).map(|&(j, _)| j).collect();

        for &j in &nib4_topk {
            let in_bert = bert_topk.contains(&j);
            if in_bert {
                bucket_a.push((i, j, nib4_dist[i][j], bert_dist[i][j]));
            } else {
                bucket_b.push((i, j, nib4_dist[i][j], bert_dist[i][j]));
            }
        }

        for &j in &bert_topk {
            if !nib4_topk.contains(&j) {
                bucket_c.push((i, j, nib4_dist[i][j], bert_dist[i][j]));
            }
        }
    }

    println!("  k={} neighborhood comparison:", k);
    println!("    Bucket A (agree):              {} pairs", bucket_a.len());
    println!("    Bucket B (nib4 close, BERT far): {} pairs  ← cadence truth", bucket_b.len());
    println!("    Bucket C (BERT close, nib4 far): {} pairs  ← surface synonymy", bucket_c.len());

    let overlap_rate = bucket_a.len() as f64 / (bucket_a.len() + bucket_b.len()) as f64;
    println!("    Overlap rate: {:.1}%", overlap_rate * 100.0);

    // Show most interesting Bucket B pairs (nib4 close, BERT far)
    // Sort by (bert_dist - nib4_dist) to find max disagreement
    let mut bucket_b_scored: Vec<_> = bucket_b.iter()
        .map(|&(i, j, nd, bd)| {
            let disagreement = bd - nd / 240.0; // normalize nib4 to ~same scale
            (i, j, nd, bd, disagreement)
        })
        .collect();
    bucket_b_scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    bucket_b_scored.dedup_by(|a, b| {
        (a.0 == b.1 && a.1 == b.0) || (a.0 == b.0 && a.1 == b.1)
    });

    println!("\n  BUCKET B — Top 20: Nib4 says close, BERT says far (cadence truth):");
    println!("    {:<40}  {:<40}  {:>6}  {:>6}", "Item A", "Item B", "Nib4d", "BERTd");
    for &(i, j, nd, bd, _) in bucket_b_scored.iter().take(20) {
        println!("    {:<40}  {:<40}  {:>6.0}  {:>6.3}",
            items[i].id, items[j].id, nd, bd);
    }

    // Show most interesting Bucket C pairs (BERT close, nib4 far)
    let mut bucket_c_scored: Vec<_> = bucket_c.iter()
        .map(|&(i, j, nd, bd)| {
            let disagreement = nd / 240.0 - bd; // normalize
            (i, j, nd, bd, disagreement)
        })
        .collect();
    bucket_c_scored.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    bucket_c_scored.dedup_by(|a, b| {
        (a.0 == b.1 && a.1 == b.0) || (a.0 == b.0 && a.1 == b.1)
    });

    println!("\n  BUCKET C — Top 20: BERT says close, Nib4 says far (surface synonymy):");
    println!("    {:<40}  {:<40}  {:>6}  {:>6}", "Item A", "Item B", "Nib4d", "BERTd");
    for &(i, j, nd, bd, _) in bucket_c_scored.iter().take(20) {
        println!("    {:<40}  {:<40}  {:>6.0}  {:>6.3}",
            items[i].id, items[j].id, nd, bd);
    }

    // ========================================================================
    // STEP 5: Per-family disagreement rates
    // ========================================================================
    println!("\n--- Step 5: Per-family disagreement rates ---\n");

    let mut families: Vec<String> = items.iter().map(|it| it.family.clone()).collect();
    families.sort();
    families.dedup();

    println!("  {:<20}  {:>5}  {:>8}  {:>8}  {:>8}  {:>6}",
        "Family", "N", "MeanTau", "MinTau", "MaxTau", "Stable?");
    for fam in &families {
        let members: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam)
            .map(|(i, _)| i)
            .collect();
        if members.is_empty() { continue; }

        let fam_taus: Vec<f64> = members.iter().map(|&i| taus[i].1).collect();
        let fam_mean = fam_taus.iter().sum::<f64>() / fam_taus.len() as f64;
        let fam_min = fam_taus.iter().cloned().fold(f64::INFINITY, f64::min);
        let fam_max = fam_taus.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let stable = if fam_mean > 0.3 { "YES" }
                    else if fam_mean > 0.15 { "~" }
                    else { "NO" };

        println!("  {:<20}  {:>5}  {:>+8.4}  {:>+8.4}  {:>+8.4}  {:>6}",
            fam, members.len(), fam_mean, fam_min, fam_max, stable);
    }

    // ========================================================================
    // STEP 6: RGB/CMYK disagreement analysis
    // ========================================================================
    println!("\n--- Step 6: RGB/CMYK mode vs BERT clustering ---\n");

    // Check if BERT clusters items by mode (RGB vs CMYK)
    let mut same_mode_bert_dists: Vec<f32> = Vec::new();
    let mut diff_mode_bert_dists: Vec<f32> = Vec::new();
    let mut same_mode_nib4_dists: Vec<f32> = Vec::new();
    let mut diff_mode_nib4_dists: Vec<f32> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if intensity_bits[i] == intensity_bits[j] {
                same_mode_bert_dists.push(bert_dist[i][j]);
                same_mode_nib4_dists.push(nib4_dist[i][j]);
            } else {
                diff_mode_bert_dists.push(bert_dist[i][j]);
                diff_mode_nib4_dists.push(nib4_dist[i][j]);
            }
        }
    }

    let mean_same_bert = same_mode_bert_dists.iter().sum::<f32>() / same_mode_bert_dists.len() as f32;
    let mean_diff_bert = diff_mode_bert_dists.iter().sum::<f32>() / diff_mode_bert_dists.len() as f32;
    let mean_same_nib4 = same_mode_nib4_dists.iter().sum::<f32>() / same_mode_nib4_dists.len() as f32;
    let mean_diff_nib4 = diff_mode_nib4_dists.iter().sum::<f32>() / diff_mode_nib4_dists.len() as f32;

    println!("  Does BERT see the RGB/CMYK (causing/caused) split?");
    println!("    BERT:  same-mode mean dist = {:.4}, diff-mode = {:.4}, ratio = {:.3}",
        mean_same_bert, mean_diff_bert, mean_diff_bert / mean_same_bert);
    println!("    Nib4:  same-mode mean dist = {:.1}, diff-mode = {:.1}, ratio = {:.3}",
        mean_same_nib4, mean_diff_nib4, mean_diff_nib4 / mean_same_nib4);
    println!();

    if (mean_diff_bert / mean_same_bert) > 1.02 {
        println!("  → BERT partially sees the mode split (ratio > 1.02)");
        println!("    This means observer language has SOME awareness of causality direction.");
    } else {
        println!("  → BERT does NOT see the mode split (ratio ≈ 1.0)");
        println!("    This means the RGB/CMYK distinction is INTERIOR ONLY.");
        println!("    Observer language cannot distinguish causing from caused.");
        println!("    This is the strongest evidence that Nib4 captures something BERT misses.");
    }

    // ========================================================================
    // STEP 7: Holes — where Nib4 predicts a cluster but language has no support
    // ========================================================================
    println!("\n--- Step 7: Holes in language (Nib4 clusters with no BERT support) ---\n");

    // Find items that are close in Nib4 space but have the highest average
    // BERT distance to their Nib4 neighbors
    let mut item_hole_scores: Vec<(usize, f64)> = Vec::with_capacity(n);

    for i in 0..n {
        // Get 5 nearest Nib4 neighbors
        let mut nib4_neighbors: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, nib4_dist[i][j]))
            .collect();
        nib4_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let top5: Vec<usize> = nib4_neighbors.iter().take(5).map(|&(j, _)| j).collect();

        // Average BERT distance to those same 5 neighbors
        let avg_bert: f64 = top5.iter().map(|&j| bert_dist[i][j] as f64).sum::<f64>() / 5.0;
        item_hole_scores.push((i, avg_bert));
    }
    item_hole_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Items whose Nib4 neighbors are BERT-distant (language holes):");
    println!("    {:<45}  {:>12}  {:>10}  {:>4}", "Item", "Family", "AvgBERTd", "Mode");
    for &(idx, score) in item_hole_scores.iter().take(20) {
        let mode = if intensity_bits[idx] { "CMYK" } else { "RGB" };
        println!("    {:<45}  {:>12}  {:>10.4}  {:>4}",
            items[idx].id, items[idx].family, score, mode);
    }

    // ========================================================================
    // VERDICT
    // ========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    QUALIA XOR VERDICT                      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                            ║");
    println!("║  Nib4  = 16 nibbles + I-bit (interior physics, CMYK)      ║");
    println!("║  BERT  = 384-dim MiniLM (observer language, RGB)           ║");
    println!("║                                                            ║");
    println!("║  Mean Kendall tau: {:>+.4}                                  ║", mean_tau);
    println!("║  Median tau:       {:>+.4}                                  ║", median_tau);
    println!("║  Overlap@k={}:     {:.1}%                                   ║", k, overlap_rate * 100.0);
    println!("║                                                            ║");
    println!("║  Bucket A (agree):        {} pairs = structural truth     ║", bucket_a.len());
    println!("║  Bucket B (Nib4 only):    {} pairs = cadence truth       ║", bucket_b.len());
    println!("║  Bucket C (BERT only):    {} pairs = surface synonymy    ║", bucket_c.len());
    println!("║                                                            ║");
    println!("║  BERT mode-split ratio: {:.3}                              ║",
        mean_diff_bert / mean_same_bert);
    println!("║  Nib4 mode-split ratio: {:.3}                              ║",
        mean_diff_nib4 / mean_same_nib4);
    println!("║                                                            ║");
    println!("║  If BERT can't see the mode split: Nib4 captures           ║");
    println!("║  interior causality that language doesn't encode.           ║");
    println!("║                                                            ║");
    println!("║  Messiness + stable sign = depth, not noise.               ║");
    println!("║                                                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
