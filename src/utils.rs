use std::cmp;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use rmp_serde::encode::Ext;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use rand::seq::SliceRandom;

pub fn minf(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

pub fn to_slices<T>(xs: &[Vec<T>]) -> Vec<&[T]> {
    xs.iter().map(|x| x.as_slice()).collect()
}

pub fn to_refs<T>(xs: &[T]) -> Vec<&T> {
    xs.iter().collect()
}

#[derive(Deserialize)]
struct WordIdsSerde {
    words: Vec<String>,
}

impl<'de> Deserialize<'de> for WordIds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let w: WordIdsSerde = Deserialize::deserialize(deserializer)?;
        let mut wids = HashMap::new();
        for (i, word) in w.words.iter().enumerate() {
            wids.insert(word.clone(), i as u32);
        }
        Ok(WordIds {
            wids: wids,
            wids_rev: w.words,
        })
    }
}

impl Serialize for WordIds {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("WordIds", 1)?;
        s.serialize_field("words", &self.wids_rev)?;
        s.end()
    }
}

pub struct WordIds {
    wids: HashMap<String, u32>,
    wids_rev: Vec<String>,
}

impl WordIds {
    pub fn new() -> Self {
        let mut wids = HashMap::new();
        wids.insert("<unk>".to_string(), 0);
        wids.insert("<s>".to_string(), 1);
        wids.insert("</s>".to_string(), 2);
        let wids_rev = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        Self {
            wids: wids,
            wids_rev: wids_rev,
        }
    }

    pub fn get_wid_mut(&mut self, word: &str) -> u32 {
        if let Some(&wid) = self.wids.get(word) {
            wid
        } else {
            let new_wid = self.wids.len() as u32;
            self.wids.insert(word.to_string(), new_wid);
            self.wids_rev.push(word.to_string());
            new_wid
        }
    }

    pub fn get_wid(&self, word: &str) -> u32 {
        if let Some(&wid) = self.wids.get(word) {
            wid
        } else {
            0
        }
    }

    pub fn get_word(&self, wid: u32) -> String {
        let wid = wid as usize;
        if let Some(word) = self.wids_rev.get(wid) {
            word.clone()
        } else {
            "<unk>".to_string()
        }
    }

    pub fn unk(&self) -> u32 {
        0
    }

    pub fn bos(&self) -> u32 {
        1
    }

    pub fn eos(&self) -> u32 {
        2
    }

    pub fn len(&self) -> u32 {
        self.wids.len() as u32
    }
}

pub fn load_corpus<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let f = BufReader::new(File::open(path)?);
    let mut corpus = vec![];
    for line in f.lines() {
        let mut sent = vec![];
        for word in line?.split_whitespace() {
            sent.push(word.to_string());
        }
        corpus.push(sent);
    }
    Ok(corpus)
}

pub fn clean_corpus<T>(
    corpus1: &Vec<Vec<T>>,
    corpus2: &Vec<Vec<T>>,
    max_len: usize,
    max_ratio: f32,
) -> (Vec<Vec<T>>, Vec<Vec<T>>)
where
    T: Clone,
{
    let mut new_corpus1: Vec<Vec<T>> = vec![];
    let mut new_corpus2 = vec![];
    for (sent1, sent2) in corpus1.iter().zip(corpus2) {
        if sent1.len() < max_len
            && sent2.len() < max_len
            && (sent1.len() as f32 / sent2.len() as f32) < max_ratio
            && (sent2.len() as f32 / sent1.len() as f32) < max_ratio
        {
            new_corpus1.push(sent1.clone());
            new_corpus2.push(sent2.clone());
        }
    }
    (new_corpus1, new_corpus2)
}

pub fn update_vocab(corpus: &Vec<Vec<String>>, size: u32, wids: &mut WordIds) {
    let mut word_count = HashMap::new();
    for sent in corpus {
        for word in sent {
            if let Some(cnt) = word_count.get_mut(word) {
                *cnt += 1;
            } else {
                word_count.insert(word.to_string(), 1);
            }
        }
    }
    let mut word_count = word_count.into_iter().collect::<Vec<(String, u32)>>();
    word_count.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (word, _) in &word_count[..cmp::min(size as usize, word_count.len())] {
        wids.get_wid_mut(word);
    }
}

pub fn make_wid_corpus(corpus: &Vec<Vec<String>>, wids: &WordIds) -> Vec<Vec<u32>> {
    let mut wid_corpus = vec![];
    for sent in corpus {
        let mut wid_sent = vec![wids.bos()];
        for word in sent {
            wid_sent.push(wids.get_wid(word));
        }
        wid_sent.push(wids.eos());
        wid_corpus.push(wid_sent);
    }
    wid_corpus
}

pub fn make_batch_itr<T>(
    src: &Vec<Vec<T>>,
    trg: &Vec<Vec<T>>,
    max_sent: usize,
    max_token: usize,
    shuffle: bool,
) -> Vec<Vec<usize>>
where
    T: Clone,
{
    let mut ids = (0..src.len()).collect::<Vec<usize>>();
    if shuffle {
        let mut rng = rand::thread_rng();
        ids.shuffle(&mut rng);
    }
    let mut result = vec![];
    let mut batch = vec![];
    let mut sent_cnt = 0;
    let mut max_sent_len = 0;
    for &id in &ids {
        sent_cnt += 1;
        let sent_len = cmp::max(src[id].len(), trg[id].len());
        max_sent_len = cmp::max(max_sent_len, sent_len);
        if sent_cnt <= max_sent && sent_cnt * max_sent_len <= max_token {
            batch.push(id);
            sent_cnt += 1;
        } else {
            result.push(batch);
            batch = vec![id];
            sent_cnt = 1;
            max_sent_len = sent_len;
        }
    }
    if batch.len() != 0 {
        result.push(batch);
    }
    result
}

pub fn make_batch(corpus: &[&[u32]], ids: &[usize], wids: &WordIds) -> Vec<Vec<u32>> {
    let max_length = ids.iter().fold(0, |acc, &i| cmp::max(acc, corpus[i].len()));
    let mut ret = vec![vec![wids.eos(); ids.len()]; max_length];
    for j in 0..ids.len() {
        for (i, &wid) in corpus[ids[j]].iter().enumerate() {
            ret[i][j] = wid;
        }
    }
    ret
}

pub fn load_msgpack<'a, T: Deserialize<'a>, P: AsRef<Path>>(filename: P) -> T {
    let mut file = File::open(filename).unwrap();
    let mut buf = vec![];
    file.read_to_end(&mut buf).unwrap();
    Deserialize::deserialize(&mut rmp_serde::Deserializer::new(&buf[..])).unwrap()
}

pub fn save_msgpack<T: Serialize, P: AsRef<Path>>(data: T, filename: P) {
    let mut buf = Vec::new();
    data.serialize(&mut rmp_serde::Serializer::new(&mut buf).with_struct_map())
        .unwrap();
    let mut file = File::create(filename).unwrap();
    file.write_all(&buf).unwrap();
}
