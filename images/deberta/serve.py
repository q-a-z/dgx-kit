"""Authenticated, single-pass DeBERTa classification using the vendor's FP32 weights."""
import hmac
import os
import threading

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

REPO = 'protectai/deberta-v3-base-prompt-injection-v2'


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)


class Predictor:
    """Keep inference in FP32 and expose the vendor's 512-token truncation explicitly."""

    def __init__(self, folder, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; refusing a silent CPU fallback')
        torch.set_num_threads(4)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32
        ).to(device).eval()
        self.lock = threading.Lock()

    def predict(self, text):
        """Return softmax labels and disclose which input tokens were evaluated."""
        with self.lock, self.torch.inference_mode():
            total = len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            probabilities = self.model(**inputs).logits.float().softmax(-1)[0].cpu().tolist()
            scores = {self.model.config.id2label[i]: probability for i, probability in enumerate(probabilities)}
            label = max(scores, key=scores.get)
            used = int(inputs['input_ids'].shape[-1])
            return {'label': label, 'score': scores[label], 'scores': scores,
                    'tokens_total': total, 'tokens_used': used, 'truncated': total > used}


def create_app(predictor, api_key):
    """Build the service with injected inference for isolated API tests."""
    if not api_key:
        raise RuntimeError('DEBERTA_API_KEY is required')
    app = FastAPI(title='DeBERTa Prompt Injection v2')

    def authorize(authorization: str | None = Header(default=None)):
        if not hmac.compare_digest(authorization or '', 'Bearer ' + api_key):
            raise HTTPException(401, 'A valid bearer key is required')

    @app.get('/health', dependencies=[Depends(authorize)])
    def health():
        return {'status': 'ok', 'loaded': [REPO], 'device': predictor.device, 'max_length': 512}

    @app.post('/v1/classify', dependencies=[Depends(authorize)])
    def classify(request: ClassificationRequest):
        return {'model': REPO, **predictor.predict(request.text)}

    return app


if __name__ == '__main__':
    import uvicorn

    predictor = Predictor('/models/deberta', os.environ.get('DEBERTA_DEVICE', 'cuda'))
    uvicorn.run(create_app(predictor, os.environ.get('DEBERTA_API_KEY', '')),
                host='0.0.0.0', port=int(os.environ.get('DEBERTA_PORT', '8203')))
