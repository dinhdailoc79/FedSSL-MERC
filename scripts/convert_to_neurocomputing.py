import os

IEEE_FILE = "paper/IEEE_TAFFC/fedssl-merc-ieee.tex"
ELS_FILE = "paper/Neurocomputing/fedssl-merc-neurocomputing.tex"

def convert():
    print(f"Reading {IEEE_FILE}...")
    with open(IEEE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Document class replacement
    content = content.replace(
        "\\documentclass[10pt,journal,compsoc]{IEEEtran}",
        "\\documentclass[final,3p,times,twocolumn]{elsarticle}"
    )

    # 2. Remove space saving tweaks
    # We find where it starts and ends
    start_tweak = content.find("% --- Space saving tweaks for page limit compliance ---")
    if start_tweak != -1:
        end_tweak = content.find("\\begin{document}")
        content = content[:start_tweak] + content[end_tweak:]

    # 3. Add elsarticle journal name
    preamble_end = "\\begin{document}"
    content = content.replace(
        preamble_end,
        "\\journal{Neurocomputing}\n\n" + preamble_end
    )

    # 4. Convert title, author, affiliations, abstract, keywords to elsarticle frontmatter format
    start_doc = content.find("\\begin{document}")
    end_intro = content.find("\\section{Introduction}")
    
    if start_doc != -1 and end_intro != -1:
        new_frontmatter = """\\begin{document}

\\begin{frontmatter}

\\title{FedSSL-MERC: Uncertainty-Aware Federated Semi-Supervised Learning for Multimodal Emotion Recognition in Conversations}

\\author[1]{Dinh Dai Loc\\corref{cor1}}
\\ead{locddse190189@fpt.edu.vn}
\\author[1]{Tran Phi Hoc}
\\ead{hoctpse190186@fpt.edu.vn}
\\author[1]{Ho Gia Phu}
\\ead{hanzopn2603@gmail.com}
\\author[1]{Le Vo Minh Thu}
\\ead{thulvm@fe.edu.vn}

\\cortext[cor1]{Corresponding author.}

\\affiliation[1]{organization={Department of Artificial Intelligence, FPT University},
            city={Ho Chi Minh City},
            country={Vietnam}}

\\begin{abstract}
Emotion Recognition in Conversations (ERC) is crucial for human-computer interaction, yet deploying ERC systems at scale faces two fundamental challenges: (1) conversational data is inherently distributed across organizations with strict privacy requirements, and (2) obtaining emotion labels is expensive, leaving most data unlabeled. We propose \\textbf{FedSSL-MERC}, a framework that unifies Evidential Deep Learning (EDL), Epistemic-Aware Federated Aggregation (EAFA), and Evidential Consistency Regularization (ECR) to address both challenges simultaneously. EDL equips each utterance with an epistemic uncertainty estimate via Dirichlet distributions, enabling EAFA to automatically down-weight unreliable client updates during aggregation. ECR replaces FixMatch's hard pseudo-labeling threshold with certainty-weighted Dirichlet KL divergence, eliminating confirmation bias in semi-supervised learning. For multimodal fusion, we employ Dempster-Shafer evidence combination, which naturally handles noisy or missing modalities by fusing at the evidence level rather than the feature level. We provide convergence guarantees showing that EAFA achieves a tighter bound than FedAvg under heterogeneous client quality. Extensive experiments across 805 configurations on MELD, IEMOCAP, and DailyDialog demonstrate that: (1) EAFA achieves 79.96\\% WF1 on IEMOCAP 4-class, obtaining highly competitive performance compared to centralized state-of-the-art baselines; (2) EAFA outperforms FedAvg under client noise while remaining highly competitive in clean settings; and (3) ECR outperforms FixMatch on MELD while avoiding pseudo-label noise.
\\end{abstract}

\\begin{keyword}
Affective Computing \\sep Emotion Recognition in Conversations \\sep Federated Learning \\sep Semi-Supervised Learning \\sep Evidential Deep Learning
\\end{keyword}

\\end{frontmatter}

"""
        content = content[:start_doc] + new_frontmatter + content[end_intro:]

    # 5. Remove IEEEPARstart macro calls
    # Let's use string replace for the specific occurrence: \IEEEPARstart{M}{ultimodal}
    content = content.replace("\\IEEEPARstart{M}{ultimodal}", "Multimodal")

    # 6. Update bibliography style
    content = content.replace(
        "\\bibliographystyle{IEEEtran}",
        "\\bibliographystyle{elsarticle-num}"
    )

    # 7. Remove biographies section at the end
    start_biography = content.find("% Biographies")
    if start_biography != -1:
        # We replace everything from % Biographies to the end with just \end{document}
        content = content[:start_biography] + "\n\\end{document}\n"
    else:
        # Fallback: find the last occurrence of \end{IEEEbiographynophoto}
        last_bio = content.rfind("\\end{IEEEbiographynophoto}")
        if last_bio != -1:
            content = content[:last_bio + len("\\end{IEEEbiographynophoto}")] + "\n\\end{document}\n"

    # Write the result
    print(f"Writing to {ELS_FILE}...")
    with open(ELS_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    convert()
