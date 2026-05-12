# LoRA Bench — Report

Modèle base : `Qwen/Qwen3-0.6B`


## Résultats GSM8K

| Étape | Accuracy [95% Wilson CI] | Tokens/s | Peak Mem (MB) | Wall (s) | N samples |
|-------|--------------------------|----------|---------------|----------|-----------|
| Baseline | 0.220 [0.168, 0.282] | 37.1 | 1137 | 2750.8 | 200 |
| SFT-LoRA | 0.340 [0.278, 0.408] | 25.0 | 1178 | 1178.2 | 200 |
| SFT+DPO | 0.370 [0.306, 0.439] | 25.0 | 1178 | 1251.5 | 200 |
| SFT+GTW | 0.370 [0.306, 0.439] | 22.6 | 1178 | 1504.5 | 200 |

## Gain absolu vs baseline

- SFT-LoRA : **+0.1200** (+12.0 pts)
- SFT+DPO : **+0.1500** (+15.0 pts)
- SFT+GTW : **+0.1500** (+15.0 pts)

## Exemples de sorties

### Baseline

**Sample 1** (gold=18, pred=3, ok=False)

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

```
<think>
Okay, let's see. Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends every day with four. Then she sells the remainder at the farmers' market for $2 per egg. The question is asking how much she makes every day at the farmers' market. Hmm, let me break this down step by step.

First, I need to figure out how many eggs she has each day. She lays 16 eggs per day. Then she eats three for breakfast. So, that's 3 eggs. Then she bakes muffins for her friends every day with four. Wait, does that mean she uses four eggs for baking? Or does she use f
```

**Sample 2** (gold=3, pred=3, ok=True)

> A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

```
<think>
Okay, let's see. The problem says a robe takes 2 bolts of blue fiber and half that much white fiber. I need to figure out how many bolts in total it takes. Hmm, let me break this down step by step.

First, the robe uses 2 bolts of blue fiber. That's straightforward. Now, they mention half that much white fiber. So, half of 2 bolts would be... let me calculate that. Half of 2 is 1. So, the white fiber takes 1 bolt. 

Wait, so if blue is 2 and white is 1, then total bolts would be 2 + 1? That would be 3. Is there anything else I need to consider here? The problem says "how many bolts in 
```

**Sample 3** (gold=70000, pred=150, ok=False)

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

```
<think>
Okay, let's see. Josh buys a house for $80,000 and then puts in $50,000 in repairs. The problem says that this increased the value of the house by 150%. I need to figure out how much profit he made. Hmm, profit... so profit is usually calculated as revenue minus cost, right? But here, he's buying the house and then putting in repairs. So maybe the profit is the difference between the selling price and the cost? Or is it the total value after repairs minus the cost? Let me think.

First, let me parse the problem again. He buys the house for $80,000. Then he puts in $50,000 in repairs. T
```

### SFT-LoRA

**Sample 1** (gold=18, pred=128, ok=False)

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

```
<think>
She eats 3 ducks x 16 eggs/duck = <<3*16=48>>48 eggs per day
She sells 16 eggs x 4 = <<16*4=64>>64 eggs per day
She makes 64 eggs x $2 = <<64*2=128>>128 dollars per day
</think>

The answer is: #### 128
```

**Sample 2** (gold=3, pred=3, ok=True)

> A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

```
<think>
A robe takes 2 bolts of blue fiber and 2/2 = <<2/2=1>>1 bolt of white fiber.
So it takes 2 + 1 = <<2+1=3>>3 bolts of fiber.
</think>

The answer is: #### 3
```

**Sample 3** (gold=70000, pred=30000, ok=False)

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

```
<think>
He increased the value of the house by 150% of $80,000, which is $80,000 * 150/100 = $<<80000*150/100=120000>>120,000
He made a profit of $80,000 - $50,000 = $<<80000-50000=30000>>30,000
</think>

The answer is: #### 30000
```

### SFT+DPO

**Sample 1** (gold=18, pred=82, ok=False)

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

```
<think>
Janet's ducks lay 16 eggs per day, so she has 16*3=<<16*3=48>>48 eggs per day.
She eats 3 eggs for breakfast, so she has 48-3=<<48-3=45>>45 eggs left.
She bakes 4 muffins for her friends, so she has 45-4=<<45-4=41>>41 eggs left.
She sells the remaining 41 eggs at $2 per egg, so she makes 41*2=<<41*2=82>>82 dollars at the farmers' market.
</think>

The answer is: #### 82
```

**Sample 2** (gold=3, pred=3, ok=True)

> A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

```
<think>
A robe takes 2 bolts of blue fiber and 2/2 = <<2/2=1>>1 bolt of white fiber.
So it takes 2 + 1 = <<2+1=3>>3 bolts of fiber.
</think>

The answer is: #### 3
```

**Sample 3** (gold=70000, pred=30000, ok=False)

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

```
<think>
He increased the value of the house by 150% of $80,000, which is $80,000 * 150/100 = $<<80000*150/100=120000>>120,000
He made a profit of $80,000 - $50,000 = $<<80000-50000=30000>>30,000
</think>

The answer is: #### 30000
```

### SFT+GTW

**Sample 1** (gold=18, pred=128, ok=False)

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

```
<think>
She eats 3 ducks x 16 eggs/duck = <<3*16=48>>48 eggs per day
She sells 16 eggs x 4 = <<16*4=64>>64 eggs per day
She makes 64 eggs x $2 = <<64*2=128>>128 dollars per day
</think>

The answer is: #### 128
```

**Sample 2** (gold=3, pred=3, ok=True)

> A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

```
<think>
A robe takes 2 bolts of blue fiber and half that much white fiber, so it takes 2/2 = <<2/2=1>>1 bolt of white fiber.
A robe takes 2 + 1 = <<2+1=3>>3 bolts of fiber.
</think>

The answer is: #### 3
```

**Sample 3** (gold=70000, pred=120000, ok=False)

> Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

```
<think>
He increased the value of the house by 150% of $80,000, which is $80,000 * 150/100 = $<<80000*150/100=120000>>120,000
He then added $50,000 to the value of the house, making it $80,000 + $120,000 = $<<80000+120000=200000>>200,000
He made a profit of $200,000 - $80,000 = $<<200000-80000=120000>>120,000
</think>

The answer is: #### 120000
```


## MATH-500 (OOD bench)

Dataset: `HuggingFaceH4/MATH-500`.

| Étape | Accuracy [95% Wilson CI] | Tokens/s | Peak Mem (MB) | Wall (s) | N samples |
|-------|--------------------------|----------|---------------|----------|-----------|
| Baseline | 0.180 [0.149, 0.216] | 39.5 | 1137 | 9715.6 | 500 |
| SFT-LoRA | 0.226 [0.192, 0.265] | 27.4 | 1178 | 6411.5 | 500 |
| SFT+DPO | 0.264 [0.227, 0.304] | 27.4 | 1178 | 6743.1 | 500 |
| SFT+GTW | 0.234 [0.199, 0.273] | 22.4 | 1178 | 8534.2 | 500 |

### Catégorisation erreurs MATH-500

| Étape | Correct | Truncation | Format | Symbolic mismatch | Reasoning |
|-------|---------|------------|--------|-------------------|-----------|
| Baseline | 90 (18%) | 3 (1%) | 2 (0%) | 108 (22%) | 297 (59%) |
| SFT-LoRA | 113 (23%) | 2 (0%) | 7 (1%) | 71 (14%) | 307 (61%) |
| SFT+DPO | 132 (26%) | 2 (0%) | 8 (2%) | 69 (14%) | 289 (58%) |
| SFT+GTW | 117 (23%) | 2 (0%) | 6 (1%) | 76 (15%) | 299 (60%) |