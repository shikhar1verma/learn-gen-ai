# Transformer Architecture Understanding

## A 6-Minute Visual Walk-Through of Transformers

*(all diagrams stitched from the screenshots you just shared)*

---

### 1. Why Language Is Tricky

> “The teacher taught the student with the book.”
> 
> 
> Who owns the book?
> 
> ![Screenshot 2025-07-23 at 00.27.26.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2Fffd76bb0-d7c6-4c21-8e56-eb2fcb99a7a9%2FScreenshot_2025-07-23_at_00.27.26.png/size/w=1350?exp=1754895910&sig=uoTbHtAySV71CPM846WmZNntGnc8ZXTWdjKrUukwLU8&id=2381f07b-36b0-805e-a9f1-c4cf14623af9&table=block&userId=68b2fc3d-8d28-47eb-a949-98a40ce68389)
> 
> Ambiguity like this is why models need **context**.
> 

---

### Simple Architecture

![Screenshot 2025-07-23 at 00.28.36.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F411aa2ab-68c3-40ab-8f1b-8029f187a7b1%2FScreenshot_2025-07-23_at_00.28.36.png/size/w=1420?exp=1754895910&sig=fDCN8uRfbSB-UwIKeN0P2If_sNBp6cJ-I749B1_q0jQ&id=2381f07b-36b0-80bc-b555-f8ec3ec8f898&table=block&userId=68b2fc3d-8d28-47eb-a949-98a40ce68389)

### 2. From Words to Numbers

First, the **Tokenizer** splits the sentence into sub-word pieces and gives each piece an ID.

![Screenshot 2025-07-23 at 00.29.06.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F1321cdcb-eaba-4dc2-83c2-dfe4228f4d57%2FScreenshot_2025-07-23_at_00.29.06.png/size/w=1420?exp=1754895910&sig=RWZjNnI6oP71WdvcW8vqXrU_vOoAGj-7IQuPKALDfL4&id=2381f07b-36b0-803a-94a6-ee37d0b7dcc7&table=block&userId=68b2fc3d-8d28-47eb-a949-98a40ce68389)

---

### 3. Embeddings + Positions

Each ID is turned into a **vector** (embedding) and a **positional code** is added so order is not lost.

![Screenshot 2025-07-23 at 00.29.31.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F11a4ec69-55b8-4f4b-b670-4fd425090d71%2FScreenshot_2025-07-23_at_00.29.31.png/size/w=1420?exp=1754897809&sig=VeIwACE4R8dHLFdS7mIykU2LkJOzXDWzrNSadQNTw9c&wasReauthorized=true)

---

![Screenshot 2025-07-23 at 00.30.12.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F0ea27e78-1d8f-4690-bfa6-239197505465%2FScreenshot_2025-07-23_at_00.30.12.png/size/w=1420?exp=1754897810&sig=FLnl8Ke3_NzINcBrBvr2U1p5z9OTcKQr_hcM5GjITl0&wasReauthorized=true)

### 4. Self-Attention in One Glance

Every token asks: *“Which other tokens should I look at?”*

The answer is a **weight matrix** (attention scores).

![Screenshot 2025-07-23 at 00.27.54.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2Fe4c434ae-c0a9-4c6d-bb10-aed1c4d309f6%2FScreenshot_2025-07-23_at_00.27.54.png/size/w=1420?exp=1754897622&sig=Quk_G_uCsdSuzRwB5N08jSvXcUsFjXVnzfoCNjzHqX8&wasReauthorized=true)

![Screenshot 2025-07-23 at 00.30.32.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F570fdaad-9efd-4199-8071-3a96923adc95%2FScreenshot_2025-07-23_at_00.30.32.png/size/w=1420?exp=1754897695&sig=c3kFLvbCeuqd2bBHruJ-tFPKj62sroGxqP-7kpruSRw&wasReauthorized=true)

> Multi-head = run the same idea in parallel “opinions”, then merge.
> 

---

### 5. One Transformer Block

Stack the following **N** times (6–96 layers depending on model size).

![Screenshot 2025-07-23 at 00.28.24.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2Fa61ac58e-ca11-4832-906a-557b1fb80ef9%2FScreenshot_2025-07-23_at_00.28.24.png/size/w=1420?exp=1754897697&sig=jIsiYDhNmRWWcwa7Lc7kuVlTsSYSlBy9TMeQ0az07qQ&wasReauthorized=true)

1. Multi-Head Attention
2. Add & Layer-Norm
3. Feed-Forward (small MLP)
4. Add & Layer-Norm again

---

### 6. Three Model Flavors

| Mode | What it does | Example |
| --- | --- | --- |
| **Encoder-only** | Understand input | BERT, sentence classification |
| **Decoder-only** | Generate text | GPT-series, Llama |
| **Encoder-Decoder** | Translate or summarize | T5, original Transformer |

![Screenshot 2025-07-23 at 00.32.23.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2Fb1f1b355-62af-4d8a-b745-f95fb6c9219b%2FScreenshot_2025-07-23_at_00.32.23.png/size/w=1420?exp=1754897699&sig=1wvZBVLx1aMkKQeM7sFH4G4pvKduAJHsUxIsNPlG8-4&wasReauthorized=true)

---

### 7. Generation Loop (Decoder-Only)

```
Prompt tokens ──► Encoder stack (context) ──► Softmax ──► Sample next token
Repeat until <EOS> token or max length.

```

![Screenshot 2025-07-23 at 00.32.10.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2Fc743666a-f53d-4308-b739-bbb1f1a76d3b%2FScreenshot_2025-07-23_at_00.32.10.png/size/w=1420?exp=1754897701&sig=-p3fagOp_KQ2ZQlF2eynbCg-N2D4MiztnlpeXaIBS2A&wasReauthorized=true)

![Screenshot 2025-07-23 at 00.32.00.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F937fe570-c2ec-478f-8d40-360ece894a92%2FScreenshot_2025-07-23_at_00.32.00.png/size/w=1420?exp=1754897703&sig=Hw6KmqqmuIJZ-vt5B-axiA7CjFFij2B5IZ5VYwrmH8E&wasReauthorized=true)

---

### 8. Geometric Intuition

Words live in a **high-dimensional space**.

Self-attention re-positions them so *semantically* related words cluster together.

![Screenshot 2025-07-23 at 00.29.43.png](https://img.notionusercontent.com/s3/prod-files-secure%2F810a7467-b0b6-4699-a1b9-d61322144746%2F8efc6fb9-9007-42b2-ab2a-fcb2f5419959%2FScreenshot_2025-07-23_at_00.29.43.png/size/w=1420?exp=1754897705&sig=z9ql2fJl-sUa9Q32dVyKsOL2evtqnzS9Hd2So0JErcY&wasReauthorized=true)

---

### 9. One-Sentence Cheat-Sheet

> Turn text into vectors, let every vector vote on every other vector through learned attention weights, stack the vote-and-update block many times, then repeatedly ask the network “what comes next?”—that’s a Transformer.
> 

---

### 10. 30-Second Recap GIF (ASCII)

```
Step-0:  [The]  → ?
Step-1:  [The teacher]  → taught
Step-2:  [The teacher taught]  → the
Step-3:  [The teacher taught the]  → student
… until completion.

```