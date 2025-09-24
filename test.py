from transformers import AutoModel, AutoConfig, AutoModelForTokenClassification, AutoModelForMaskedLM, AutoTokenizer
# model_name = "/users/dwils152/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/25abaf0bd247444fcfa837109f12088114898d98"
# model = AutoModel.from_pretrained(
#             model_name, trust_remote_code=True)
# print(model)

# tok = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", trust_remote_code=True)

# model = AutoModelForTokenClassification.from_pretrained(
#             "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
#             num_labels=1,
#             trust_remote_code=True)

# seq = "TATATACCCCCCNNNAT<eos>TANNATTTGCGATCACACCCCCC"

# inputs = tok(seq, return_tensors="pt")
# print(inputs)
# print(tok.get_vocab())
# #outputs = model(**inputs)
# #print(outputs.logits[:, 1:, :])

from transformers import AutoConfig, AutoModelForCausalLM

model_name = 'togethercomputer/evo-1-8k-base'

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision="1.1_fix")
model_config.use_cache = True

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=model_config,
    trust_remote_code=True,
    revision="1.1_fix"
)