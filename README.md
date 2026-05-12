# The Fake-News Simulator That Learned to Ask Who Was Reading
*By Claire Yuqing Yang*

Fake news spreads through people with histories, habits, anxieties, and tastes. A retired voter who distrusts platforms, a college student who loves dramatic stories, and a politically moderate parent scrolling between errands may all see the same post and react differently. This project builds a misinformation simulator around that simple idea: a good model of diffusion should pay attention to the reader as much as to the rumor.

I use the public Twin-2K digital-twin dataset as the human backbone for the simulation. The dataset contains 2,058 real U.S. survey participants with rich demographic, psychological, cognitive, economic, personality, and behavioral measures. I turn those participant summaries into agent personas, then use `gpt-4.1-nano` as a generative-AI layer inside an agent-based model of fake-news diffusion. The aim is practical: test where LLM-based personas add value to a policy simulator, and where transparent ABM structure still carries the causal work.

The first step is a project-specific sanity check. The Twin-2K paper already benchmarks digital twins broadly; here I check my own downstream pipeline. I use the `story_beliefs` split, where real participants read story chapters and rated valence, arousal, and interest in continuing. For 60 participants, I give `gpt-4.1-nano` the participant's persona summary plus the same chapter, then ask it to predict that participant's answers. This tells me which human response dimensions my exact prompt, model, and data slice capture before I use the personas in a misinformation setting.

![Digital twin validation](outputs/digital_twin_validation.png)  
*Figure 1. Project-specific digital-twin check on 60 real Twin-2K participants. The LLM twin predicts arousal better than a population-mean baseline (MAE 0.83 vs. 1.24), while valence and interest remain harder targets.*

The validation result is useful because it is specific. Arousal is the clearest win: 83% of arousal predictions land within one point of the human answer. Valence and interest are less reliable in this setup. That pattern gives the rest of the project a disciplined foundation. The personas bring real individual-level texture, and the model earns its role most clearly when predicting affective intensity.

I then place those same real-persona summaries into a fake-news diffusion model. I compare three designs. `pure_abm` uses hand-written behavioral rules over traits derived from the Twin-2K summaries. `pure_llm` asks the LLM for a full behavior profile: belief probability, sharing propensity, belief lability, and resistance to correction. `abm_llm` gives the LLM a narrower job: estimate how important, emotional, and personally relevant each story feels to each participant-persona. The ABM handles exposure, network diffusion, interventions, and counterfactual changes.

![Counterfactual stability across model variants](outputs/cf_stability_box.png)  
*Figure 2. Mean paired counterfactual response in fake-news reach using 60 real Twin-2K personas as the agent backbone. Dots show repeated runs; the cached pure-LLM design stays flat, while the hybrid design gives trait changes an explicit path through the ABM.*

The counterfactual experiment is the core policy test. I shift media literacy and fake-story emotionality, then measure how fake-news reach changes under the same random seeds. The hybrid model responds most visibly when fake emotionality decreases: fake reach falls by about 6 percentage points on average. The pure LLM-persona line is best read as a diagnostic for one design choice: caching a full persona-story behavior profile before the perturbation. A pure LLM simulator could be re-queried for each counterfactual, but then the estimated effect would combine the intervention with a newly generated behavioral profile. The hybrid design gives the LLM a narrower perception task and leaves the policy lever inside a manipulable ABM mechanism.

![Cascade-size heavy tails by model variant](outputs/heavytail_ccdf.png)  
*Figure 3. Cascade-size CCDFs under the baseline policy using the Twin-2K persona backbone. The plot compares whether simulated fake and true cascades separate in the heavy tail.*

The final chart asks whether the simulation also recovers a famous empirical signature. Vosoughi, Roy, and Aral found that falsehoods on Twitter traveled farther, faster, and in more extreme cascades than truth. My closed 60-person simulation works at a classroom scale, so I treat this as a calibration check for the direction of the pattern. In these runs, true-news cascades are at least as large as fake-news cascades across the main variants; for the hybrid model, the mean fake cascade is 5.13 agents versus 6.89 for true news. That result gives the op-ed its larger point: digital twins make agents more grounded, while outcome validation still decides how far the simulation can travel.

The project’s takeaway is a division of labor. Real digital-twin personas give misinformation simulation a stronger empirical starting point. Generative AI is most helpful here as a perception engine that translates rich participant summaries into story-specific reactions. The ABM supplies the part social science needs for policy: explicit mechanisms, interventions, and counterfactuals that can be moved and inspected. Together, they form a simulator that asks a better question every time a rumor appears: who is reading this, and what would change their response?

Source data: [Twin-2K-500 public dataset](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), [Twin-2K-500 Mega Study](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study), [Vosoughi, Roy & Aral 2018](https://www.science.org/doi/10.1126/science.aap9559)

Analysis outputs: [digital twin validation metrics](data/digital_twin_validation_metrics.csv), [OpenAI persona predictions](data/digital_twin_story_predictions.csv), [counterfactual summary](outputs/cf_stability_summary.csv), [cascade tail summary](outputs/heavytail_tail_summary.csv)
