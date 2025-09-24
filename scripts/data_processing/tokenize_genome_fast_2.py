import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

_manager = None
_global_counter = None

def init_manager():
    """
    Create the global manager/counter exactly once in the main process.
    We'll store it in module-level globals so that workers can import it.
    """
    global _manager, _global_counter
    _manager = Manager()
    _global_counter = _manager.Value('i', 0)  # integer

def reserve_chunk_ids(num_chunks: int):
    """
    Atomically reserve a block of chunk IDs in the global counter,
    returning the start offset.
    """
    with _global_counter.get_lock():
        start_id = _global_counter.value
        _global_counter.value += num_chunks
    return start_id

def _process_one_chrom(chrom_id, seq_str, model_name, chunk_size):
    # 1) Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_ids = tokenizer(seq_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze()

    total_tokens = input_ids.shape[0]
    num_chunks = total_tokens // chunk_size
    if num_chunks == 0:
        return [], []

    # 2) Reserve unique chunk IDs for this chromosome
    start_id = reserve_chunk_ids(num_chunks)

    # 3) Chunking / writing
    input_ids = input_ids[: num_chunks * chunk_size]
    chunks = torch.split(input_ids, chunk_size)

    fasta_lines = []
    bed_lines   = []

    global_start = 0
    chunk_id = start_id

    for chunk in chunks:
        tokens = tokenizer.convert_ids_to_tokens(chunk)
        tokens_cleaned = [t.replace(" ", "") for t in tokens]
        joined_tokens  = "".join(tokens_cleaned)
        chunk_length   = len(joined_tokens)

        start = global_start
        stop  = start + chunk_length

        # chunk_id is globally unique now
        fasta_lines.append(f">{chrom_id}:chunk_{chunk_id:07d}:{start}-{stop}\n{joined_tokens}\n")

        local_pos = start
        for t in tokens_cleaned:
            tlen = len(t)
            bed_lines.append(f"{chrom_id}\t{local_pos}\t{local_pos + tlen}\t{chunk_id:07d}\n")
            local_pos += tlen

        global_start = stop
        chunk_id += 1

    return fasta_lines, bed_lines

def main(fasta_in, model_name, chunk_size, workers):
    init_manager()  # set up global manager/counter

    with ExitStack() as stack:
        fasta_out = stack.enter_context(open("my_chunks.fa", 'w'))
        bed_out   = stack.enter_context(open("my_chunks.bed", 'w'))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for record in SeqIO.parse(fasta_in, "fasta"):
                future = executor.submit(
                    _process_one_chrom,
                    record.id,
                    str(record.seq),
                    model_name,
                    chunk_size
                )
                futures.append(future)

            for future in as_completed(futures):
                fasta_lines, bed_lines = future.result()
                fasta_out.write(''.join(fasta_lines))
                bed_out.write(''.join(bed_lines))
