# RepoCoder Pipeline (Extended)

This project is inspired by [Microsoft’s RepoCoder](https://github.com/microsoft/CodeT) — a retrieval-augmented framework for repository-level code completion.

> **Citation**  
>
> ```bibtex
> @article{zhang2023repocoder,
>   title={RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation},
>   author={Zhang, Fengji and Chen, Bei and Zhang, Yue and Liu, Jin and Zan, Daoguang and Mao, Yi and Lou, Jian-Guang and Chen, Weizhu},
>   journal={arXiv preprint arXiv:2303.12570},
>   year={2023}
> }
> ```

This extended version provides a modular, fully-automated pipeline for evaluating repository-level code completion with the RepoCoder methodology. The pipeline supports three key retrieval paradigms:

- **RG1 (Retrieve-and-Generate):** Uses top-k retrieval from the repository as context for completing a masked target.
- **GT (Ground Truth):** Uses the oracle context centered around the target as a reference baseline.
- **RG-RG (RepoCoder):** A second round of retrieval using model predictions inserted into the source to regenerate prompts and improve completions.

## Conceptual Overview

The goal is to evaluate how retrieval can enhance long-range code completion at the repository level. The process revolves around constructing and manipulating **context windows**—chunks of code extracted from repositories—which are later used to retrieve similar code segments and construct prompts for language models.

### Step 1: Repository Windowing

To build a retrieval corpus, the pipeline first slices every Python file in each repository into **sliding context windows**. A window is a fixed number of lines (e.g., 20), sampled every few lines (e.g., every 10 lines). These windows are saved with metadata (file name, line numbers, etc.) and used as the searchable database for later retrieval.

This windowing is performed **agnostic to any task**, purely to index the repo structure.

Purpose:

- Create a dense, searchable corpus of code chunks for similarity retrieval.
- Represent all possible surrounding contexts in the codebase.

### Step 2: Vectorization

Each context window is then converted into a **vector representation**. This repo supports:

- **Bag-of-Words (1-gram)**: A simple token-based representation based on the number of unigrams.
- (Future extensibility exists for embedding models like `text-embedding-ada-002`.)

Purpose:

- Translate raw code into a numerical format that allows similarity comparison.
- Facilitate fast nearest-neighbor search during retrieval.

These vectors are saved and reused for each retrieval strategy.

### Step 3: Task-Based Windowing (RG1 and GT)

A benchmark file (e.g., `short_api_benchmark`) defines **specific locations in the code** where a completion should be evaluated. For each location:

- **RG1 (Retrieve-and-Generate)** extracts a one-sided window that ends at the target line. This simulates a real-world scenario where the model completes code after reading what came before.
  
- **GT (Ground Truth)** extracts a symmetric window centered on the target line, treating it as a reference for comparison. This reflects an ideal oracle context.

Purpose:

- RG1 serves as the **query** for retrieval.
- GT serves as the **gold standard** for performance evaluation.

These task-specific windows are later used to build prompts.

### Step 4: Retrieval

Using the vectorized task windows (from RG1 and GT), the pipeline performs **nearest-neighbor search** against the repository-wide window vectors.

- Top-k similar code fragments are retrieved.
- These fragments act as **in-context examples** for the prompt.

Purpose:

- Leverage structurally and semantically similar code as reference material for the model.
- Emulate how developers might "look around the repo" for patterns.

### Step 5: Prompt Construction

The retrieved fragments are formatted as **commented code blocks** with file path annotations and included above the original prompt.

A prompt typically looks like:

```bash
the below code fragment can be found in:
utils.py
--------------------------------------------------
def load_json(file_path):
with open(file_path) as f:
return json.load(f)
--------------------------------------------------
Here is some code to complete...
```

- Length is constrained by model context size (e.g., 2048 tokens).
- Multiple fragments are included if space allows.

**Purpose:**

- Mimic few-shot examples using real repository context.
- Improve generation by grounding the model in relevant patterns.

### Step 6: Inference (External to This Pipeline)

The `.jsonl` files of prompts are passed to a model such as **CodeGen**, **Codex**, or **GPT-4**.

- The model generates completions conditioned on the prompt.
- Output format should include the generated `choices` and match the benchmark structure.

This step must be run separately using the model of your choice.

### Step 7: Prediction-Based Retrieval (RepoCoder / RG-RG)

This is the **key innovation** introduced by RepoCoder:

- After inference, model predictions are inserted back into the source code at the original locations.
- New context windows are sliced around these predictions.
- These windows are vectorized and used to **re-run retrieval** and **rebuild prompts** (as in Steps 2–5).

This "retrieve-then-generate-then-retrieve-again" loop allows the model to refine its completion by attending to **context that was not visible during the first generation**.

Purpose:

- Boost completion quality with second-stage retrieval tailored to the model's own prediction.
- Simulate an iterative reasoning process.

## Execution Flow in `run.py`

The main driver script runs the following steps:

```python
make_repo_windows(...)            # Step 1: Build and vectorize full-repo sliding windows

run_rg1_and_gt_stage(...)         # Step 2–5: Build RG1 and GT windows, retrieve, construct prompts

run_repocoder_stage(...)          # Step 6–7: Insert predictions, re-vectorize, re-retrieve, rebuild prompts
```

These steps output:

- Vector files (.pkl)
- Retrieval results (.pkl)
- Final prompts for inference (.jsonl)

From there, evaluation proceeds by running completions and scoring them against ground truth.

Steps:

### Run `python run.py`

#### Run `run_repo_stage()` to generate windows and vectorize all repositories

- Make repo windows via `MakeWindowWrapper().window_for_repo_files()` inside `RepoWindowMaker` (source here runs `_build_windows_for_file()`)
  - Slice step: slice_step = max(1, window_size // slice_size) = max(1, 20 // 2) = 10
  - Iterate through all lines via: `range(0, len(code_lines), slice_step)`, e.g., every 10 lines
  - `start_line = max(0, line_no - slice_step)`
  - `end_line = min(len(code_lines), line_no + window_size - slice_step)`
- Merge same context windows by appending metadata to a list of metadata and the same context via `RepoWindowMaker.build_windows() -> RepoWindowMaker._merge_windows_with_same_context()`
- Vectorize all context windows and store at `data/cache/window/repos/{repo}_ws{window_size}_slice{slice_size}.pkl` and `data/cache/vector/repos/{repo}_ws{window_size}_slice{slice_size}.pkl`

#### Run `run_rg1_and_gt_stage()` for Short API Benchmark

- Inside `run_rg1_and_gt_stage()`, run `make_baseline_and_ground_windows()` for benchmark, all repos, all window sizes, all slice sizes
  - Inside `make_baesline_and_ground_windows()`, run `MakeWindowWrapper().window_for_baseline_and_ground()` to make windows for repos for Baseline and Ground Truth. For each window size, slice size, and repo:
    - Run `BaselineWindowMaker().build_window()`
      - Metadata included in a Task:
        - `prompt`
        - `metadata.task_id`
        - `metadata.ground_truth`
        - `metadata.fpath_tuple`
        - `metadata.context_start_lineno` # start of context. With (implicit) `context_end_lineno = line_no - 1`
        - `metadata.line_no` # line where the ground truth answer exists
      - `start_line = max(context_start_lineno, line_no - self.window_size)`
        - Note: with start_line, it's likely were not pulling in the full prompt, but only including one window size from the target line number
      - end_line = line_no, with the last line being excluded when retrieving the context
      - Saved contexts as dictionary pickle file via `create_search_window_path()`. This is the RG1 method
    - Run `GroundTruthWindowMaker().build_window()`
      - Same metadata as `BaselineWindowMaker()`
      - `start_line = max(context_start_lineno, line_no - self.delta_size)` where `self.delta_size = window_size // 2`, so we are including even less code than the BaselineWindowMaker approach
      - `end_line = line_no + self.window_size - self.delta_size`, but here we are including more code past the line number. Clamps to `min(len(code_lines), end_line)` if end_line is past end of code file.
      - Example: `BaselineWindowMaker`
        - context_start_lineno: 20
        - line_no: 57
        - start_line: max(context_start_lineno = 20, line_no - window_size = 57 - 20 = 37) = 37
        - end_line: line_no = 57
        - Extract code from lines 37-56 (inclusive, 20 lines)
      - Example: `GroundTruthWindowMaker`
        - context_start_lineno: 20
        - line_no: 57
        - start_line: max(context_start_lineno = 20, line_no - delta_size = 57 - 10 = 47) = 47
        - end_line: line_no + window_size - delta_size = 57 + 20 - 10 = 67
        - Extract code from lines 47-66 (inclusive, 20 lines)
        - Store output at: `data/cache/winodws/{benchmark}/{mode}/{repo}_ws{window_size}_slice{slice_size}.pkl"`
          - Benchmarks: `short_api`, `short_line`, `random_api`, `random_line`
          - Mode: `gt` (ground truth), `r-g` (retrieve-and-generate), `r-g-r-g` (two stage RAG, RepoCoder)
- Inside `run_rg1_and_gt_stage()`, run `vectorize_baseline_and_ground_windows()` which runs `BuildVectorWrapper().vectorize_baseline_and_ground_windows()` for a benchmark, iterating over all repos, window sizes, and slice sizes
  - Generates embeddings for both baseline and ground truth windows for all particular tasks of a benchmark
  - Store output at: `data/cache/vector/{benchmark}/{mode}/{repo}_ws{window_size}_slice{slice_size}.one-gram.pkl"`. Difference is swap `window` with `vector` and add `one-gram.pkl` extension for BagOfWords vectorizer
- Inside `run_rg1_and_gt_stage()`, run `search_baseline_and_ground()` which runs `CodeSearchWrapper().search_baseline_and_ground()` for a benchmark, iterating over all repos, window sizes, and slice sizes
  - Create GT and RG1 search window paths via `_run_parallel()`
    - Pulls in all vectors of a given repo (`repo_embedding_path`) via: `data/cache/vector/repos/huggingface_diffusers_ws20_slice2.one-gram.pkl`
    - Pulls in all vectors of a repo's tasks (`query_line_path`) via: `data/cache/vector/short_api/r-g/huggingface_diffusers_ws20_slice2.one-gram.pkl`
    - Uses these `query_line_path` and `repo_embedding_path` to find and rank similar vector embeddings via `CodeSearchWorker.run()`.
      - In `run()`, iterate through all lines in `query_line_path`, check against all lines in repo embedding path, and compute a similarity score for the repo embedding line (every line) vs. query line embedding.  Only compute scores if the repo embedding line occurs BEFORE the query line embedding.  If it occurs later in the file, do not include or compute the score.
      - Then, add the repo embedding line to `top_k_context`.
      - Sort the `top_k_context` and only include the `max_top_k` entries with the highest similarity scores.
    - Output retrieval results at `data/cache/retrieval/{benchmark}/{mode}/{repo}_ws{window_size}_slice{slice_size}.one-gram.top{max_top_k}.pkl`, note the difference is `retrieval` path, and one-gram vectorization technique, and `max_top_k`. This output has a new field called `top_k_context` which includes the repo embedding lines and the similarity score. This entire process is run on both the GT and RG1 task windows. We will, at most, include `k` retrieved contexts for a given query context.
- Inside `run_rg1_and_gt_stage()`, run `build_prompts_for_baseline_and_ground()` which runs `BuildPromptWrapper().build_first_search_prompt()` for all benchmarks (RG1 and GT), slice sizes, and window sizes
  - Inside `BuildPromptWrapper().build_first_search_prompt()`, run `self._run()` to generate a prompt
    - Inside `self._run()`, iterate through all repos and generate prompts based on the retrieval file found at `data/cache/retrieval/{benchmark}/{mode}/{repo}_ws{window_size}_slice{slice_size}.one-gram.top{max_top_k}.pkl`. Run `build_2nd_stage_input_file()` inside `self._run()` and store all of the prompts per repo in a `lines` list. A prompt includes the following information:

    ```bash
    {
        "prompt": prompt,
        "metadata": {
            **task["metadata"],
            "query_window": {
                "context": query["context"],
                "metadata": query["metadata"],
            },
            "top_k_context": [
                {
                    "context": c[0]["context"],
                    "metadata": c[0]["metadata"],
                    "sim_score": c[1],
                }
                for c in context
            ],
            "window_size": query["metadata"]["window_size"],
            "slice_size": (
                context[0][0]["metadata"][0]["slice_size"] if context else None
            ),
        },
    }
    ```

    - Essentially, each repo contains a list of prompts to run, and these prompts are all concatenated together.
  - Prompt output is stored at: `cache/prompts/{mode}-{vector_type}-ws-{window_size}-ss-{slice_size}.jsonl"`
