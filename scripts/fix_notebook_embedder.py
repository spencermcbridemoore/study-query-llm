#!/usr/bin/env python3
"""Fix the notebook embedder to filter empty texts."""

import json
import sys
import os
from pathlib import Path

# Change to script directory
script_dir = Path(__file__).parent.parent
os.chdir(script_dir)

notebook_path = Path("notebooks/pca_kllmeans_sweep.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell with _embed_batch_async
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "_embed_batch_async" in source and "async def _embed_batch_async" in source:
            # Replace the function
            new_source = '''    async def _embed_batch_async(texts: list[str]) -> np.ndarray:
        """Async wrapper for embedding."""
        # #region agent log
        import json
        from datetime import datetime, timezone
        try:
            with open(r'c:\\Users\\spenc\\Cursor Repos\\study-query-llm\\.cursor\\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"location": "notebook:_embed_batch_async:entry", "message": "Embedding batch entry", "data": {"n_texts": len(texts), "deployment": deployment, "text_lengths": [len(t) if t else 0 for t in texts], "empty_texts": [i for i, t in enumerate(texts) if not t or not t.strip()]}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000), "runId": "debug-run", "hypothesisId": "A"}) + '\\n')
        except Exception:
            pass
        # #endregion
        
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            service = EmbeddingService(repository=repo)

            # Filter out empty texts before creating requests
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            # #region agent log
            try:
                with open(r'c:\\Users\\spenc\\Cursor Repos\\study-query-llm\\.cursor\\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"location": "notebook:_embed_batch_async:after_filter", "message": "After filtering empty texts", "data": {"original_count": len(texts), "valid_count": len(valid_texts), "filtered_indices": [i for i in range(len(texts)) if i not in valid_indices]}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000), "runId": "debug-run", "hypothesisId": "B"}) + '\\n')
            except Exception:
                pass
            # #endregion

            # Create embedding requests only for valid texts
            requests = [
                EmbeddingRequest(text=text, deployment=deployment)
                for text in valid_texts
            ]

            # Get embeddings (will use cache if available)
            responses = await service.get_embeddings_batch(requests)

            # Extract vectors
            embeddings = [resp.vector for resp in responses]
            
            # #region agent log
            try:
                with open(r'c:\\Users\\spenc\\Cursor Repos\\study-query-llm\\.cursor\\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"location": "notebook:_embed_batch_async:exit", "message": "Embedding batch exit", "data": {"n_responses": len(responses), "n_embeddings": len(embeddings)}, "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000), "runId": "debug-run", "hypothesisId": "C"}) + '\\n')
            except Exception:
                pass
            # #endregion
            
            # Pad embeddings for filtered texts with zeros (maintain same length as input)
            if len(valid_indices) < len(texts):
                # Create full embedding array with zeros for filtered texts
                full_embeddings = np.zeros((len(texts), embeddings[0].shape[0] if embeddings else 0), dtype=np.float64)
                for idx, emb in zip(valid_indices, embeddings):
                    full_embeddings[idx] = emb
                return full_embeddings
            
            return np.asarray(embeddings, dtype=np.float64)'''
            
            # Find where the function starts and ends in the source
            lines = source.split('\n')
            new_lines = []
            in_function = False
            indent_level = 0
            skip_until_end = False
            
            for line in lines:
                if "async def _embed_batch_async" in line:
                    in_function = True
                    indent_level = len(line) - len(line.lstrip())
                    # Replace with new function
                    new_lines.append(new_source)
                    skip_until_end = True
                    continue
                elif skip_until_end:
                    # Check if we've reached the end of the function (next def or class at same or less indent)
                    stripped = line.lstrip()
                    if stripped and not line.startswith(' ' * (indent_level + 1)) and not line.startswith('\t'):
                        if not (stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''")):
                            skip_until_end = False
                            new_lines.append(line)
                    # Otherwise skip this line (it's part of the old function)
                    continue
                else:
                    new_lines.append(line)
            
            cell["source"] = new_lines
            print(f"Updated cell {i}")
            break

# Save the notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully")
