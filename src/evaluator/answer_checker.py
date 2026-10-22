def genrater(self, promt):

    print("Index:", self.index_pt)
    print("Chunks:", self.chunks_pt)

    # -------------------------
    # 1. Initial retrieval
    # -------------------------

    result = reviewer(
        promt,
        self.index_pt,
        self.chunks_pt,
        8
    )

    print("Initial retrieval:", result)

    # -------------------------
    # 2. Initial reranking
    # -------------------------

    result = rerank(
        promt,
        result,
        top_k=5
    )

    print("Initial reranking:", result)

    # -------------------------
    # 3. Evaluate retrieval
    # -------------------------

    status = evaluate_retrieval(
        result,
        0.3
    )

    # -------------------------
    # 4. CRAG corrective retrieval
    # -------------------------

    if status == "bad":

        print("Chunks were not good.")
        print("Running corrective retrieval...")

        result = corrective_reranking(
            promt,
            self.index_pt,
            self.chunks_pt,
            4
        )

        if result == "bad":

            print("Corrective retrieval also failed.")

            return {
                "answer": "I couldn't find enough relevant information in the uploaded document.",
                "source": "document"
            }

        print("Corrective retrieval succeeded.")

    else:

        print("Initial retrieval was good.")

    # -------------------------
    # 5. Build context
    # -------------------------

    context = build_context(result)

    print("We got context:")
    print(context)

    # -------------------------
    # 6. Generate + Self-RAG
    # -------------------------

    answer = generate_answer(
        promt,
        context
    )

    print("Generated answer:", answer)

    return {
        "answer": answer,
        "source": "document"
    }