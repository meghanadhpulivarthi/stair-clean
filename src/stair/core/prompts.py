SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with selecting the most relevant "
    "sections from a book's table of contents that best answers a user query."
)

USER_PROMPT = (
    "Book: {title}\n"
    "Select all relevant sections from the table of contents below that can "
    "help answer the user query. Return the output only as a Python list of "
    "strings, where each string follows the format:\n"
    '"section_num title"\n'
    'Example Output: ["1.1 Section Name", "2.3 Another Section"]\n'
    "Do not include any explanations or additional text.\n\n"
    "Table of Contents:\n{toc}\n\n"
    "Query:\n{question}\n\n"
    "Relevant Sections:"
)
