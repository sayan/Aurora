Below is a step-by-step guide to help you:

1. **Install and Import the Necessary Libraries**  
   You’ll need to install the following (if you haven’t already):  
   - `datasets` (Hugging Face’s library for datasets)  
   - `transformers` (Hugging Face’s library for models, tokenizers, etc.)  
   - `torch` (PyTorch)  
   - `torchvision` (sometimes helpful, but not absolutely required if you’re focusing on text)  

   ```bash
   pip install datasets transformers torch
   ```

   **Imports**:
   ```python
   from datasets import load_dataset
   from transformers import AutoTokenizer, AutoModel
   import torch
   import torch.nn as nn
   from torch.utils.data import DataLoader, Dataset
   ```

2. **Load a Dataset using Hugging Face `datasets`**  
   The `datasets` library provides a convenient `load_dataset` function to load many popular NLP datasets.  
   - For example, let’s load the [IMDb dataset](https://huggingface.co/datasets/imdb), which is a sentiment classification dataset.

   ```python
   dataset = load_dataset("imdb")  # This returns a dictionary-like object with 'train' and 'test' splits
   print(dataset)
   ```
   You’ll typically see something like:
   ```
   DatasetDict({
       train: Dataset({
           features: ['text', 'label'],
           num_rows: 25000
       })
       test: Dataset({
           features: ['text', 'label'],
           num_rows: 25000
       })
   })
   ```

3. **Choose and Load a Model & Tokenizer**  
   Using Hugging Face Transformers, you often load a pre-trained model and corresponding tokenizer. For example, we can load a pretrained [BERT-base-uncased](https://huggingface.co/bert-base-uncased):

   ```python
   model_name = "bert-base-uncased"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModel.from_pretrained(model_name)
   ```

   **Note**: If you want a model that is suitable for classification (like `BertForSequenceClassification`), you could use that instead of `AutoModel`. For demonstration, we’ll keep it general with `AutoModel`.

4. **Preprocess the Dataset with the Tokenizer**  
   You must tokenize the raw texts so that they become valid model inputs.  
   - We typically apply the tokenizer on each text sample.  
   - The tokenizer will return a dictionary with (by default) `input_ids` and `attention_mask`.  
   - We will also keep the labels.

   ```python
   def tokenize_function(examples):
       return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

   tokenized_dataset = dataset.map(tokenize_function, batched=True)
   ```
   This will apply `tokenize_function` to each example in your dataset. It adds new fields like `"input_ids"` and `"attention_mask"` to each example.  
   After tokenization, you might see something like:
   ```
   DatasetDict({
       train: Dataset({
           features: ['text', 'label', 'input_ids', 'attention_mask'],
           num_rows: 25000
       })
       ...
   })
   ```

5. **Convert Dataset Splits to PyTorch-friendly Format**  
   The Hugging Face `Dataset` object can be converted to PyTorch tensors using `set_format`, or you can create a custom `Dataset` subclass. A common approach is:

   ```python
   tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
   ```
   Now, each item of the dataset will be a dictionary containing `input_ids`, `attention_mask`, and `label` as PyTorch tensors (if your dataset has labels with name `"label"`).

6. **Create a PyTorch DataLoader**  
   You can wrap the dataset in a `DataLoader` to easily iterate in batches.

   ```python
   train_dataset = tokenized_dataset["train"]
   test_dataset = tokenized_dataset["test"]

   train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
   ```
   - `batch_size` can be adjusted based on your GPU/CPU memory.  
   - `shuffle=True` for the training set is typical.

7. **Wrap the Model inside a Custom `nn.Module` (Optional)**  
   If you want more flexibility, you can wrap your Hugging Face model inside your own PyTorch `nn.Module`. This is useful if you want to add custom layers on top. For a straightforward pass-through, you can do:

   ```python
   class CustomModel(nn.Module):
       def __init__(self, pretrained_model_name):
           super(CustomModel, self).__init__()
           self.bert = AutoModel.from_pretrained(pretrained_model_name)
           # Example: Additional layer
           self.classifier = nn.Linear(768, 2)  # If BERT hidden size is 768, and we have 2 classes

       def forward(self, input_ids, attention_mask):
           # Extract the last hidden states from BERT
           outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
           # Typically, the last hidden state is at outputs.last_hidden_state (batch_size, seq_len, hidden_dim)
           # If we want the [CLS] token representation, we can take outputs.last_hidden_state[:,0,:]
           cls_output = outputs.last_hidden_state[:, 0, :]  # shape (batch_size, hidden_dim)
           # Pass it through a classifier (for classification tasks)
           logits = self.classifier(cls_output)
           return logits

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   custom_model = CustomModel(model_name).to(device)
   ```

8. **Forward Pass with the DataLoader**  
   Now that we have our `DataLoader` and our custom model, we can do a forward pass. Usually, you’ll do something like this inside a training loop or evaluation loop:

   ```python
   # Example of just one batch
   batch = next(iter(train_loader))
   input_ids = batch["input_ids"].to(device)
   attention_mask = batch["attention_mask"].to(device)
   labels = batch["label"].to(device)

   # Forward pass
   outputs = custom_model(input_ids, attention_mask)
   print("Logits shape:", outputs.shape)  # Expect (batch_size, num_classes) for classification
   ```

   If you want to compute a loss (e.g., cross-entropy for classification):

   ```python
   criterion = nn.CrossEntropyLoss()
   loss = criterion(outputs, labels)
   print("Loss:", loss.item())
   ```

9. **(Optional) Training Loop Skeleton**  
   If you want to go further and see how you might train, here’s a minimal structure:

   ```python
   optimizer = torch.optim.AdamW(custom_model.parameters(), lr=1e-5)
   criterion = nn.CrossEntropyLoss()

   epochs = 1

   for epoch in range(epochs):
       custom_model.train()
       total_loss = 0

       for batch in train_loader:
           optimizer.zero_grad()
           input_ids = batch["input_ids"].to(device)
           attention_mask = batch["attention_mask"].to(device)
           labels = batch["label"].to(device)

           outputs = custom_model(input_ids, attention_mask)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()

           total_loss += loss.item()

       avg_loss = total_loss / len(train_loader)
       print(f"Epoch {epoch+1}, Loss: {avg_loss}")

       # Optionally evaluate on test set
       # custom_model.eval()
       # ... (evaluation code)
   ```

10. **Summary**  
   - **Step 1**: Install and import libraries.  
   - **Step 2**: Load dataset via `load_dataset`.  
   - **Step 3**: Load model + tokenizer (e.g., `AutoTokenizer`, `AutoModel`).  
   - **Step 4**: Tokenize the dataset (via `.map`).  
   - **Step 5**: Convert dataset to torch format (`set_format`).  
   - **Step 6**: Use `DataLoader` for batching.  
   - **Step 7**: Optionally wrap the Hugging Face model in a custom PyTorch `nn.Module`.  
   - **Step 8**: Perform forward passes by iterating over the `DataLoader`.  
   - **Step 9**: (Optional) Set up a training loop to optimize model parameters.  

That’s the general outline of using Hugging Face datasets, tokenizers, and models in a PyTorch training pipeline. Feel free to experiment by swapping in other models (e.g., `AutoModelForSequenceClassification`) and other datasets.