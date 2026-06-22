import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set font family to sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def draw_framework():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color palette
    c_input = '#E8F5E9'       # Soft Green
    c_model = '#E3F2FD'       # Soft Blue
    c_fed = '#EDE7F6'         # Soft Purple
    c_output = '#FFF3E0'      # Soft Orange
    c_reliability = '#FCE4EC' # Soft Pink

    # Border colors
    b_input = '#2E7D32'
    b_model = '#1565C0'
    b_fed = '#6A1B9A'
    b_output = '#EF6C00'
    b_reliability = '#C2185B'

    # Helper function to draw rounded boxes with text
    def draw_box(x, y, w, h, text, bg_color, border_color, fontsize=10, fontweight='normal'):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.5,
            mutation_scale=0.4
        )
        ax.add_patch(rect)
        # Add text centered
        ax.text(
            x + w/2, y + h/2, text,
            color='#1A1A1A',
            fontsize=fontsize,
            fontweight=fontweight,
            ha='center',
            va='center',
            multialignment='center'
        )

    # 1. Inputs (Left)
    ax.text(1.2, 7.3, "1. Multimodal Inputs", fontsize=11, fontweight='bold', color=b_input)
    draw_box(0.5, 5.8, 1.8, 1.0, "Text Modality\n(RoBERTa-Base)", c_input, b_input, fontweight='bold')
    draw_box(0.5, 4.3, 1.8, 1.0, "Audio Modality\n(WavLM-Base)", c_input, b_input, fontweight='bold')
    draw_box(0.5, 2.8, 1.8, 1.0, "Contextual\nDialogue History", c_input, b_input)

    # 2. Client Model (Middle-Left)
    ax.text(4.2, 7.3, "2. Client Local Training", fontsize=11, fontweight='bold', color=b_model)
    draw_box(3.5, 4.0, 2.5, 2.5, "EDL-DialogueRNN\n\n- Modality Late Fusion\n- Dirichlet Prior Head\n- Evidential Consistency\nRegularization (ECR)", c_model, b_model, fontweight='bold')

    # 3. Federated Server Aggregation (Middle)
    ax.text(7.7, 7.3, "3. FedSSL-MERC Server", fontsize=11, fontweight='bold', color=b_fed)
    draw_box(7.2, 4.2, 2.2, 2.0, "EAFA Aggregation\n\nServer averages\nweights dynamically using\nclient uncertainty\nattenuation $w_k^{\\mathrm{EAFA}}$", c_fed, b_fed, fontweight='bold')

    # 4. Outputs (Middle-Right)
    ax.text(10.7, 7.3, "4. Evidential Outputs", fontsize=11, fontweight='bold', color=b_output)
    draw_box(10.4, 5.3, 1.6, 1.0, "Probability $p_c$\n(Class Probabilities)", c_output, b_output, fontweight='bold')
    draw_box(10.4, 3.7, 1.6, 1.0, "Vacuity $u$\n(Epistemic\nUncertainty)", c_output, b_output, fontweight='bold')

    # 5. Reliability Branches (Right)
    ax.text(13.2, 7.3, "5. Reliability Modules", fontsize=11, fontweight='bold', color=b_reliability, ha='center')
    draw_box(12.3, 5.5, 2.2, 1.2, "RQ1 & RQ3\nSelective Prediction\n(Abstain if $u > \\tau$)", c_reliability, b_reliability, fontweight='bold')
    draw_box(12.3, 3.8, 2.2, 1.2, "RQ2 & RQ3\nConformal Prediction\n(LAC / RAPS sets)", c_reliability, b_reliability, fontweight='bold')
    draw_box(12.3, 2.1, 2.2, 1.2, "RQ4 (Appendix D)\nOOD Detection\n(Speaker Hold-out)", c_reliability, b_reliability, fontweight='bold')

    # Connecting Arrows
    arrow_props = dict(arrowstyle="->", lw=1.5, color='#424242', mutation_scale=15)
    double_arrow_props = dict(arrowstyle="<->", lw=1.8, color='#5E35B1', mutation_scale=15)

    # Inputs -> Model
    ax.annotate("", xy=(3.3, 6.3), xytext=(2.5, 6.3), arrowprops=arrow_props)
    ax.annotate("", xy=(3.3, 4.8), xytext=(2.5, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(3.3, 3.3), xytext=(2.5, 3.3), arrowprops=arrow_props)

    # Model <-> Fed Aggregation (Loop)
    ax.annotate("Local Params\n$\\theta_k$", xy=(7.0, 5.5), xytext=(6.2, 5.5), arrowprops=arrow_props, ha='center', fontsize=8)
    ax.annotate("Global Params\n$\\theta^t$", xy=(6.2, 4.7), xytext=(7.0, 4.7), arrowprops=arrow_props, ha='center', fontsize=8)

    # Fed Aggregation -> Outputs
    ax.annotate("", xy=(10.2, 5.8), xytext=(9.6, 5.2), arrowprops=arrow_props)
    ax.annotate("", xy=(10.2, 4.2), xytext=(9.6, 5.2), arrowprops=arrow_props)

    # Outputs -> Reliability Modules
    # Prob & Vacuity -> Selective
    ax.annotate("", xy=(12.1, 6.1), xytext=(12.1, 6.1), arrowprops=arrow_props)
    ax.plot([12.1, 12.1], [5.8, 6.1], color='#424242', lw=1.5)
    ax.annotate("", xy=(12.1, 6.1), xytext=(11.9, 6.1), arrowprops=arrow_props)
    
    # Simple straight connectors for visual clarity
    # Probability and Vacuity mapping to Selective, Conformal, OOD
    ax.annotate("", xy=(12.1, 6.1), xytext=(12.1, 5.8), arrowprops=arrow_props)
    
    # We will draw clean, custom line paths for the branches:
    # Top output to selective and conformal
    ax.plot([12.1, 12.1], [5.8, 6.1], color='#424242', lw=1.5)
    # Let's draw arrows from outputs to reliability
    ax.annotate("", xy=(12.1, 6.1), xytext=(12.05, 6.1), arrowprops=arrow_props)
    
    # Connect Probability (10.4, 5.3) to Selective (12.3, 5.5) and Conformal (12.3, 3.8)
    ax.annotate("", xy=(12.1, 6.1), xytext=(12.0, 5.8), arrowprops=arrow_props)
    
    # Let's use simple direct arrows from the outputs area to the reliability area
    ax.annotate("", xy=(12.1, 6.1), xytext=(12.05, 6.1), arrowprops=arrow_props)
    
    # Clean up connectors:
    # 1. Prob (12.0, 5.8) -> Selective (12.3, 6.1)
    ax.annotate("", xy=(12.1, 6.1), xytext=(12.0, 5.8), arrowprops=arrow_props)
    # 2. Prob & Vacuity -> Conformal
    ax.annotate("", xy=(12.1, 4.4), xytext=(12.0, 4.2), arrowprops=arrow_props)
    # 3. Vacuity -> OOD
    ax.annotate("", xy=(12.1, 2.7), xytext=(12.0, 4.2), arrowprops=arrow_props)

    # Let's draw explicit connection lines from outputs
    # For Probability:
    ax.plot([12.0, 12.1], [5.8, 5.8], color='#424242', lw=1.5)
    ax.annotate("", xy=(12.15, 6.1), xytext=(12.0, 5.8), arrowprops=arrow_props)
    ax.annotate("", xy=(12.15, 4.4), xytext=(12.0, 5.8), arrowprops=arrow_props)
    
    # For Vacuity:
    ax.annotate("", xy=(12.15, 4.4), xytext=(12.0, 4.2), arrowprops=arrow_props)
    ax.annotate("", xy=(12.15, 2.7), xytext=(12.0, 4.2), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig('paper/Neurocomputing/framework_overview.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Framework overview diagram updated successfully!")

if __name__ == '__main__':
    draw_framework()
