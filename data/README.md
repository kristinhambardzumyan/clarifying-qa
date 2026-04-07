## Data Format Comparison

CoQA-ABG data is aligned with the AmbigQA/NQ schema by merging context (story + history + current turn) into a single `question` field and mapping clarification fields to the same answer-response structure. Only ambiguous CoQA-ABG examples are used in this reformatted version.

### AmbigQA (reference format)

```json
{
  "id": "-5325827848883062343",
  "question": "Who sings with every beat of my heart?",
  "nq_answers": ["Taylor Dayne"],
  "isambig": true,
  "answers": ["The Royals", "Gladys Knight & The Pips", "James Brown"],
  "clarification": {
    "question": "Are you asking about the artist who originally recorded the song \"Every Beat of My Heart\" or a cover version?",
    "answers": [
      {
        "answer": "The original artist.",
        "response": "Gladys Knight & The Pips"
      },
      {
        "answer": "A cover version.",
        "response": "The Royals"
      }
    ]
  }
}
```

### Original CoQA-ABG
```json
{
  "id": "3ftf2t8wlri896r0rn6xpwffosj9we|6|2",
  "story": "...",
  "history_turns": [...],
  "target_turn": {
    "question": "Where was it located?",
    "answer": "the east side"
  },
  "clarification_turn": {
    "question": "Do you mean Buda or Pest?",
    "answers": [
      {
        "clr_ans": "Pest",
        "org_ans": "the east side of the river"
      },
      {
        "clr_ans": "Buda",
        "org_ans": "the west side of the river"
      }
    ]
  }
}
```

### Converted (CoQA-ABG → AmbigQA-style)
```json
{
  "id": "3ftf2t8wlri896r0rn6xpwffosj9we|6|2",
  "question": "Story: ...\n\nConversation history:\nQ: ...\nA: ...\n\nCurrent question:\nWhere was it located?",
  "nq_answers": ["the east side"],
  "isambig": true,
  "answers": [
    "the east side of the river",
    "the west side of the river"
  ],
  "clarification": {
    "question": "Do you mean Buda or Pest?",
    "answers": [
      {
        "answer": "Pest",
        "response": "the east side of the river"
      },
      {
        "answer": "Buda",
        "response": "the west side of the river"
      }
    ]
  }
}
