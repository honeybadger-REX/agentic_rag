def build_context(results):
    """
    Combine retrieved chunks into one context.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        text = result["text"]

        context_parts.append(
            f"[Document {i}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return context