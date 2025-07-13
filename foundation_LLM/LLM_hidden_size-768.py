#!pip install datasets transformers --upgrade

import os, torch, torch.nn as nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import IterableDataset
from itertools import islice
from torch.cuda.amp import autocast, GradScaler
from google.colab import drive

drive.mount('/content/drive')
torch.set_float32_matmul_precision('high')

CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
CHECKPOINT_INTERVAL = 1000
MAX_STEPS = 200000
ACCUM_STEPS = 8
EVAL_PROMPTS = ["The future of AI is", "In a world where machines can think,", "The quick brown fox"]

# Model & Norm definitions (unchanged)...

class SmolLM2Config:
    vocab_size = 49152
    hidden_size = 768
    intermediate_size = 3072
    num_attention_heads = 8
    num_hidden_layers = 30
    max_position_embeddings = 2048
    hidden_act = "silu"
    rms_norm_eps = 1e-5
    dtype = torch.float16

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.weight * x / (norm + self.eps)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.ln1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.ln2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, attn_mask=None):
        x_resid = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x_resid + attn_output
        x = x + self.ff(self.ln2(x))
        return x

class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 🔁 Weight tying: share weights between embed and head
        #self.head.weight = self.embed.weight

    def forward(self, input_ids):
        input_ids = input_ids.long()
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T).long()
        x = self.embed(input_ids) + self.position_embed(pos_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50):
        input_ids = input_ids.long()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

class TokenizedStreamDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=2048):
        self.dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._printed = False

    def __iter__(self):
        buffer = ""
        for example in self.dataset:
            if not self._printed:
                print("🔍 Sample raw text:", example["text"][:200].replace("\n","⏎"))
                self._printed = True
            buffer += example["text"] + "\n"
            tokens = self.tokenizer(buffer, return_tensors=None, truncation=False)["input_ids"]
            while len(tokens) >= self.max_length:
                chunk = tokens[:self.max_length]
                yield torch.tensor(chunk)
                tokens = tokens[self.max_length:]
                buffer = self.tokenizer.decode(tokens)
def save_checkpoint(model, optimizer, scheduler, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}.pt"))
    print(f"✅ Checkpoint saved at step {step}")

def load_latest_checkpoint(model, optimizer, scheduler):
    if not os.path.exists(CHECKPOINT_DIR):
        return 0
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_step")]
    if not ckpts:
        return 0
    latest = max(ckpts, key=lambda x: int(x.replace("checkpoint_step", "").replace(".pt", "")))
    state = torch.load(os.path.join(CHECKPOINT_DIR, latest))
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    print(f"🔁 Resumed from checkpoint {latest}")
    return state["step"]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    model = SmolLM2(SmolLM2Config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=MAX_STEPS)
    scaler = GradScaler()
    start_step = 0
    # (load_checkpoint omitted for brevity)
    start_step = load_latest_checkpoint(model, optimizer, scheduler)
    model.train()
    accum_loss = 0.0
    optimizer.zero_grad()

    dataset = TokenizedStreamDataset(tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for step, batch in enumerate(islice(dataloader, start_step, MAX_STEPS), start=start_step):
        batch = batch.to(device)
        targets = batch.clone()

        if step == 0:
            print("🔍 Input IDs:", batch[0, :10].tolist())
            print("🔍 Target IDs:", targets[0, :10].tolist())

        with autocast(dtype=torch.float16):
            logits = model(batch)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id  # OK even when pad_token==eos
            )
            #print("logits dtype:", logits.dtype, "targets dtype:", targets.dtype)
            #print("logits max/min:", logits.max().item(), logits.min().item())


        scaler.scale(loss / ACCUM_STEPS).backward()
        accum_loss += loss.item()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            avg_loss = accum_loss / ACCUM_STEPS
            print(f"[Step {step}] average loss: {avg_loss:.4f}")
            accum_loss = 0.0

        if step % CHECKPOINT_INTERVAL == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step)
            model.eval()
            with torch.no_grad():
                for prompt in EVAL_PROMPTS:
                    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    out = model.generate(prompt_ids, max_new_tokens=50)
                    print(f"Prompt: {prompt}\n→ {tokenizer.decode(out[0], skip_special_tokens=True)}\n")
            model.train()

if __name__ == "__main__":
    train()
