from pathlib import Path

from stair.core.data.build_pdf import write_bookmarked_pdf


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "example_book" / "source.pdf"

CHAPTERS = [
    (
        "Introduction",
        [
            "This guide is a short, original introduction to baking sourdough bread at home. "
            "It was written for this project as a small, clearly structured example corpus.",
            "Each chapter below covers one part of the process, from building a starter to "
            "troubleshooting a loaf that did not rise.",
        ],
    ),
    (
        "Building a Starter",
        [
            "A sourdough starter is a mixture of flour and water that captures wild yeast and "
            "bacteria from the air and the flour itself.",
            "To build one, mix equal parts flour and water in a jar, and feed it with the same "
            "amounts once a day for about a week, discarding half before each feeding.",
            "A healthy starter roughly doubles in size a few hours after feeding, and smells "
            "pleasantly sour rather than harsh or like nail polish remover.",
        ],
    ),
    (
        "Mixing the Dough",
        [
            "Combine flour, water, salt, and a portion of active starter in a large bowl.",
            "Rest the dough for thirty minutes so the flour fully absorbs the water, a step "
            "called autolyse, before kneading or folding it further.",
            "Fold the dough over itself every thirty minutes for the first two hours to build "
            "strength without traditional kneading.",
        ],
    ),
    (
        "Shaping and Proofing",
        [
            "Once the dough has roughly doubled in bulk, turn it out and shape it into a tight "
            "round or oval loaf.",
            "Place the shaped loaf in a floured basket and let it proof, either at room "
            "temperature for a few hours or in the refrigerator overnight for more flavor.",
        ],
    ),
    (
        "Baking",
        [
            "Preheat a covered pot in the oven to a high temperature, then carefully place the "
            "proofed loaf inside and score the top with a blade.",
            "Bake covered for the first part of the bake to trap steam, then uncover to let the "
            "crust brown for the remainder.",
        ],
    ),
    (
        "Troubleshooting",
        [
            "A dense, flat loaf usually means the starter was not active enough, or the dough "
            "was underproofed before baking.",
            "A loaf that spreads out flat instead of holding its shape usually means the dough "
            "was overproofed or shaped too loosely.",
            "A pale crust that never browns usually means the oven was not preheated to a high "
            "enough temperature before baking.",
        ],
    ),
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_bookmarked_pdf(OUTPUT_PATH, CHAPTERS)
    print(f"Example corpus written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
