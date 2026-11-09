# Data exploration summary

## 1. Sizes and integrity

- train: 202,689 tokens, 12,439 sentences, IDs 1..202689
- test: 25,094 tokens, 2,077 sentences, IDs 500000..525093
- sample: 227,783 rows, columns ['ID', 'Label']
- sample IDs == train IDs + test IDs: True
- sentence_id overlap train/test: 0
- train: token_id runs 1..n in every sentence: True
- test: token_id runs 1..n in every sentence: True
- train: root count per sentence: {1: 12439}  (key = number of roots)
- train: sentences with heads outside 0..n or self-loops: 0
- train: sentences with cycles: 0
- train: sentences with words that cannot reach the root: 0
- train: valid trees: 12439 / 12439  (later tree statistics use only these)
- test sentences whose exact text also appears in train: 96 (4.6%); tokens covered: 0.7%

## 2. Sentence lengths


### Sentence length (tokens)

|       |    train |    test |
|:------|---------:|--------:|
| count | 12439    | 2077    |
| mean  |    16.29 |   12.08 |
| std   |    12.38 |   10.61 |
| min   |     1    |    1    |
| 25%   |     7    |    4    |
| 50%   |    14    |    9    |
| 75%   |    22    |   17    |
| max   |   159    |   81    |

### Length buckets (share of sentences and of tokens)

|       |   train sents % |   test sents % |   train tokens % |   test tokens % |
|:------|----------------:|---------------:|-----------------:|----------------:|
| 1-5   |            19.3 |           31   |              3.3 |             7.1 |
| 6-10  |            17.9 |           25.1 |              8.8 |            16.3 |
| 11-20 |            33   |           26.5 |             30.8 |            32.5 |
| 21-30 |            18.2 |           11.2 |             27.6 |            23.2 |
| 31-50 |            10   |            5.4 |             22.9 |            17   |
| 51+   |             1.7 |            0.8 |              6.5 |             3.9 |

## 3. Dependency relations

- 51 relation labels, 36 base labels (before ':')

### Relation frequency

| rel           |   % of tokens |
|:--------------|--------------:|
| punct         |         11.51 |
| case          |          8.54 |
| nsubj         |          7.8  |
| det           |          7.7  |
| root          |          6.14 |
| advmod        |          5.5  |
| obj           |          4.84 |
| amod          |          4.7  |
| obl           |          4.31 |
| conj          |          3.73 |
| compound      |          3.52 |
| mark          |          3.48 |
| nmod          |          3.36 |
| cc            |          3.31 |
| aux           |          3.17 |
| cop           |          2.31 |
| advcl         |          1.86 |
| nmod:poss     |          1.8  |
| xcomp         |          1.51 |
| ccomp         |          1.03 |
| acl:relcl     |          0.94 |
| flat          |          0.91 |
| parataxis     |          0.75 |
| acl           |          0.73 |
| nummod        |          0.72 |
| appos         |          0.68 |
| aux:pass      |          0.66 |
| nsubj:pass    |          0.58 |
| nmod:unmarked |          0.55 |
| obl:unmarked  |          0.49 |
| discourse     |          0.4  |
| compound:prt  |          0.36 |
| list          |          0.33 |
| iobj          |          0.32 |
| expl          |          0.3  |
| fixed         |          0.26 |
| nmod:desc     |          0.17 |
| obl:agent     |          0.15 |
| csubj         |          0.14 |
| nsubj:outer   |          0.11 |
| det:predet    |          0.09 |
| vocative      |          0.07 |
| goeswith      |          0.06 |
| advcl:relcl   |          0.06 |
| cc:preconj    |          0.04 |
| reparandum    |          0.02 |
| orphan        |          0.01 |
| csubj:pass    |          0    |
| dislocated    |          0    |
| dep           |          0    |
| csubj:outer   |          0    |

- top 10 relations cover 64.8% of tokens
- relations with < 100 examples: ['cc:preconj', 'reparandum', 'orphan', 'csubj:pass', 'dislocated', 'dep', 'csubj:outer']

### Top 15 relations: % of tokens within each sentence-length bucket

| rel      |   1-5 |   6-10 |   11-20 |   21-30 |   31-50 |   51+ |
|:---------|------:|-------:|--------:|--------:|--------:|------:|
| punct    |  20.4 |   15.4 |    11.5 |    10.4 |    10.3 |  10.9 |
| case     |   2.6 |    6.2 |     8.1 |     9.1 |     9.8 |  10   |
| nsubj    |   4.2 |    9.3 |     8.6 |     7.8 |     7   |   6.7 |
| det      |   2.2 |    6.4 |     7.4 |     8.1 |     8.6 |   8.6 |
| root     |  35.5 |   12.5 |     6.6 |     4   |     2.7 |   1.6 |
| advmod   |   3   |    6.2 |     5.8 |     5.6 |     5.2 |   5.2 |
| obj      |   2.6 |    5.1 |     5.2 |     4.9 |     4.6 |   4.8 |
| amod     |   3.9 |    3.7 |     4.3 |     4.8 |     5.3 |   5.8 |
| obl      |   0.8 |    3.4 |     4.4 |     4.5 |     4.8 |   4.6 |
| conj     |   0.7 |    1.9 |     3.4 |     4.1 |     4.5 |   5.2 |
| compound |   2.8 |    3.1 |     3.4 |     3.7 |     3.7 |   3.4 |
| mark     |   0.2 |    1.9 |     3.3 |     3.8 |     4.2 |   3.9 |
| nmod     |   1.1 |    2.2 |     2.9 |     3.7 |     4   |   4.5 |
| cc       |   0.9 |    2.1 |     3.2 |     3.6 |     3.8 |   4   |
| aux      |   1.2 |    3.4 |     3.4 |     3.4 |     2.9 |   2.7 |

## 4. Head distance and direction

- head is the adjacent word: 39.1%
- head within 2 words: 62.9%
- head within 5 words: 86.7%
- head within 10 words: 94.3%
- head to the right of the word: 61.7%

### Per relation: direction and distance

| rel           |   count |   head_right_pct |   median_absdist |   mean_absdist |
|:--------------|--------:|-----------------:|-----------------:|---------------:|
| punct         |   23335 |             33   |              4   |            7.6 |
| case          |   17315 |             95.7 |              2   |            2   |
| nsubj         |   15808 |             96.3 |              2   |            2.6 |
| det           |   15597 |            100   |              1   |            1.6 |
| advmod        |   11145 |             76.4 |              1   |            2.1 |
| obj           |    9802 |              3.5 |              2   |            2.2 |
| amod          |    9520 |             97.5 |              1   |            1.4 |
| obl           |    8735 |             10.9 |              4   |            4.7 |
| conj          |    7557 |              0   |              5   |            7   |
| compound      |    7138 |             99.4 |              1   |            1.3 |
| mark          |    7053 |            100   |              1   |            2.5 |
| nmod          |    6817 |              1.1 |              3   |            3.5 |
| cc            |    6700 |            100   |              2   |            2.6 |
| aux           |    6420 |             99.6 |              2   |            1.7 |
| cop           |    4683 |             96.3 |              2   |            2.3 |
| advcl         |    3763 |             25.1 |              6   |            7   |
| nmod:poss     |    3658 |            100   |              1   |            1.6 |
| xcomp         |    3055 |              0.2 |              2   |            2.4 |
| ccomp         |    2088 |              4.9 |              4   |            5.7 |
| acl:relcl     |    1901 |              0   |              3   |            4.2 |
| flat          |    1847 |              0   |              1   |            1.4 |
| parataxis     |    1525 |              4.3 |              9   |           11.3 |
| acl           |    1487 |              0.7 |              2   |            2.9 |
| nummod        |    1464 |             82.9 |              1   |            1.4 |
| appos         |    1383 |              0   |              3   |            3.7 |
| aux:pass      |    1343 |             97.6 |              1   |            1.2 |
| nsubj:pass    |    1171 |             97.3 |              3   |            3.8 |
| nmod:unmarked |    1120 |             24.7 |              1   |            1.7 |
| obl:unmarked  |     994 |             46.1 |              2   |            3.4 |
| discourse     |     802 |             80.8 |              3   |            4.3 |
| compound:prt  |     736 |              0   |              1   |            1.3 |
| list          |     665 |              0   |              4   |            6.7 |
| iobj          |     642 |              0   |              1   |            1.2 |
| expl          |     598 |             90   |              1   |            1.6 |
| fixed         |     519 |              0   |              1   |            1.1 |
| nmod:desc     |     353 |             87.8 |              1   |            1.2 |
| obl:agent     |     312 |              0.3 |              3   |            3.9 |
| csubj         |     275 |             32.7 |              4   |            5.3 |
| nsubj:outer   |     213 |            100   |              7   |            7.9 |
| det:predet    |     174 |            100   |              2   |            2.4 |
| vocative      |     137 |             40.9 |              3   |            3.6 |
| goeswith      |     123 |              0   |              1   |            1.3 |
| advcl:relcl   |     119 |              0   |              4   |            5.5 |
| cc:preconj    |      79 |            100   |              1   |            2.2 |
| reparandum    |      36 |             88.9 |              2   |            2.3 |
| orphan        |      26 |             15.4 |              2.5 |            3   |
| csubj:pass    |       7 |             28.6 |              6   |            7.3 |
| dislocated    |       5 |            100   |              4   |           11.2 |
| dep           |       3 |              0   |              4   |            3.3 |
| csubj:outer   |       2 |            100   |              7.5 |            7.5 |

- root relative position in sentence (0=start, 1=end): median 0.17

### Most common root words

| lower   |   count |
|:--------|--------:|
| have    |     302 |
| thanks  |     174 |
| stated  |     169 |
| had     |     141 |
| believe |     123 |
| let     |     118 |
| know    |      98 |
| get     |      92 |
| need    |      84 |
| go      |      84 |
| is      |      79 |
| desire  |      77 |
| has     |      74 |
| $       |      66 |
| like    |      64 |

## 5. Tree shape

- non-projective arcs: 0.16% of tokens
- sentences with at least one non-projective arc: 2.29%
  (low values mean Eisner, a projective decoder, loses almost nothing versus Chu-Liu-Edmonds)
- tree depth: median 4, max 17
- dependents per head (among words that have any): mean 2.69, max 18

## 6. Tokens and vocabulary

- train vocabulary (lowercased): 16,496 types
- test tokens unseen in train: 7.5%
- test types unseen in train: 29.9%
- (preview) OOV rate of a random 10% sentence split of train: 4.4%  <- should be close to the test rate

### Token shape

| token       |   train % |   test % |
|:------------|----------:|---------:|
| lower       |      72   |     68.5 |
| Capitalised |      12.8 |     14.9 |
| punct       |      11.8 |     12.8 |
| ALLCAPS     |       1.7 |      1.9 |
| number      |       1.3 |      1.3 |
| mixed       |       0.4 |      0.5 |
| empty       |       0.1 |      0.1 |

- if every word got its most frequent relation (seen in train, scored on train): 71.2% relation accuracy (an optimistic upper bound for word identity alone)

### Most ambiguous frequent words

| lower   |   count |   n_relations | top_relations              |
|:--------|--------:|--------------:|:---------------------------|
| it      |    1829 |            23 | nsubj, obj, expl           |
| one     |     475 |            22 | nummod, nsubj, obj         |
| all     |     516 |            20 | det, det:predet, advmod    |
| that    |    2013 |            20 | mark, nsubj, det           |
| out     |     441 |            19 | compound:prt, advmod, case |
| way     |     205 |            19 | obl, obj, obl:unmarked     |
| state   |     200 |            18 | root, compound, xcomp      |
| in      |    3080 |            18 | case, mark, advmod         |
| a       |    3758 |            18 | det, case, compound        |
| up      |     389 |            17 | compound:prt, advmod, case |
| $       |     251 |            17 | root, obj, compound        |
| what    |     431 |            17 | obj, nsubj, obl            |
| s       |    1070 |            16 | case, cop, aux             |
| time    |     391 |            16 | obl, obj, obl:unmarked     |
| like    |     407 |            16 | case, mark, root           |

## 7. Rare and unseen words


### Validation tokens by how often the word was seen in training

| freq_bucket   |   tokens_pct |   word_majority_rel_acc |   median_head_dist |
|:--------------|-------------:|------------------------:|-------------------:|
| unseen        |          4.4 |                     0.5 |                  2 |
| 1-2           |          5.1 |                    36.3 |                  2 |
| 3-10          |          9.5 |                    38.3 |                  2 |
| 11-100        |         21.8 |                    46.6 |                  2 |
| 100+          |         59.2 |                    80.4 |                  2 |

### Relations of unseen words vs all words

| rel      |   unseen words % |   all words % |
|:---------|-----------------:|--------------:|
| conj     |             11.8 |           3.8 |
| root     |             10.8 |           6   |
| compound |              9.6 |           3.6 |
| amod     |              8.8 |           4.8 |
| obj      |              8.2 |           5   |
| nmod     |              7.4 |           3.3 |
| obl      |              6.7 |           4.3 |
| flat     |              5.6 |           0.9 |
| nsubj    |              5.1 |           7.9 |
| list     |              3.9 |           0.4 |
| xcomp    |              2.8 |           1.5 |
| advcl    |              2.8 |           1.9 |

- unseen words are mostly content words (names, nouns, adjectives), so word identity can't help them: suffix and shape features must

## 8. Suffixes and prefixes


### How well one affix predicts the relation of unseen words (n=768)

| feature                          |   rel acc on unseen alphabetic words % |
|:---------------------------------|---------------------------------------:|
| suffix len 1                     |                                    6.9 |
| suffix len 2                     |                                   16.4 |
| suffix len 3                     |                                   19   |
| suffix len 4                     |                                   14.5 |
| prefix len 1                     |                                    4   |
| prefix len 2                     |                                    6.2 |
| prefix len 3                     |                                    8.2 |
| prefix len 4                     |                                    7.2 |
| always predict 'case' (baseline) |                                    0.1 |

### Common 3-letter suffixes of words with 5+ letters (at least 20 different words each)

| suf   |   count |   word_types | top_relations        |   top_% |
|:------|--------:|-------------:|:---------------------|--------:|
| ing   |    4432 |          948 | advcl, root, conj    |    17.5 |
| ion   |    1637 |          376 | obj, obl, nmod       |    19.4 |
| ted   |    1259 |          281 | root, amod, conj     |    34.3 |
| ent   |     978 |          166 | amod, obj, nmod      |    14.5 |
| ter   |     855 |          109 | case, compound, obj  |    14.6 |
| ers   |     710 |          241 | nsubj, nmod, obj     |    21.4 |
| nce   |     678 |          105 | obj, compound, obl   |    16.2 |
| ate   |     666 |          120 | root, amod, compound |    20.3 |
| her   |     642 |           36 | amod, det, advmod    |    35   |
| ons   |     607 |          161 | obj, nmod, obl       |    25.2 |
| lly   |     549 |          111 | advmod, root, conj   |    90.2 |
| ice   |     510 |           23 | obj, obl, compound   |    16.1 |
| ble   |     509 |          132 | amod, root, conj     |    34.2 |
| ies   |     504 |          147 | obj, nmod, conj      |    25.8 |
| ght   |     482 |           46 | aux, advmod, amod    |    15.1 |

## 9. Train vs test statistics


### Corpus statistics

|                                    |   train (all) |   val preview (vs train part) |   test (vs all train) |
|:-----------------------------------|--------------:|------------------------------:|----------------------:|
| sentences                          |      12439    |                       1243    |               2077    |
| tokens                             |     202689    |                      20694    |              25094    |
| mean sentence length               |         16.29 |                         16.65 |                 12.08 |
| % one-token sentences              |          4.11 |                          2.98 |                  7.27 |
| mean token length (chars)          |          4.09 |                          4.11 |                  4.13 |
| % punctuation tokens               |         11.85 |                         11.48 |                 12.76 |
| % capitalised tokens               |         12.77 |                         12.56 |                 14.9  |
| % numbers                          |          1.26 |                          1.46 |                  1.32 |
| % sentences starting capitalised   |         87.06 |                         86.65 |                 79.1  |
| % sentences ending in punctuation  |         86.55 |                         87.29 |                 76.89 |
| % tokens unseen in reference vocab |        nan    |                          4.43 |                  7.48 |

## 7. Punctuation

- punct is 11.5% of train tokens; every one of them counts for the score
- punct tokens whose shape is all-symbols: 99.1%
- all-symbol tokens labelled punct: 96.3%