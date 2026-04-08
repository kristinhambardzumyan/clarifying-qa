# Clarifying Question Generation (SFT)

This project implements Supervised Fine-Tuning (SFT) for generating clarifying questions from ambiguous queries.  

---

## Results

| Experiment | Dataset | BLEU | ROUGE-L |
|-----------|--------|------|---------|
| Exp 1 | AmbigQA | 0.083 | 0.266 |
| Exp 2 | CoQA-ABG | 0.020 | 0.123 |

---

## Training Summary

| Experiment | Training Time (min) | Final Train Loss | Steps | Samples/sec |
|-----------|--------------------|------------------|-------|-------------|
| Exp 1 (AmbigQA) | 120.25 | 0.618 | 1500 | 1.66 |
| Exp 2 (CoQA-ABG) | 25.85 | 0.947 | 276 | 1.43 |

---

## Example Predictions

### AmbigQA

**Q:** Who sings with every beat of my heart?  
**Ground Truth (Clarify Q):** Are you asking about the artist who originally recorded the song or a cover version?    
**Prediction:** The original artist of the song "Every Beat of My Heart" is Bobby Womack, though there are also cover versions.

---

### CoQA-ABG

**Q:** What did they remove?  
**Ground Truth (Clarify Q):** What did they remove from the object or system?   
**Prediction:** Do you mean what they removed to get free or something else in the story?
