import matplotlib
import matplotlib.pyplot as plt
import numpy as np

""" IMPRESSION
frame-level:

    Interprt-img      &  0.0336  &  0.3359 &  0.0270 &  0.2014 &  0.2709 &  0.1738
    senet             &  0.1678  &  0.2776 &  0.0538 &  0.0299 &  0.3093 &  0.1510
    hrnet             &  0.2175  &  0.2998 & -0.0039 &  0.1680 &  0.1945 &  0.1752
    swin-transformer  & -0.0273  &  0.0470 &  0.0361 & -0.0142 &  0.0860 &  0.0256
    mean             0.0979	0.240075	0.02825	0.096275	0.215175	0.135535

 
        
segment-level:

    ResNet            & -0.0502  &  0.0065 &  0.0191 &  0.0406 &  0.0003 &  0.0032    
    3d-resnet         & -0.0478  &  0.0102 &  0.0478 &  0.0499 & -0.0240 &  0.0072
    slow-fast         & -0.0102  &  0.0076 &  0.0010 & -0.0063 &  0.0161 &  0.0016
    tpn               &  0.0448  &  0.0348 &  0.0287 & -0.0177 & -0.0281 &  0.0125
    vat               & -0.0139  & -0.0016 &  0.0016 & -0.0013 & -0.0006 & -0.0031
    mean              -0.01546	0.0115	0.01964	0.01304	-0.00726	0.004292

        
video-level:

    3d-resnet         & -0.017 &  0.0076 &  0.0914 &  -0.0304 &  -0.047 &  0.001  
    slow-fast         & -0.0078 &  0.0133 &  0.01 &  0.0048 &  -0.0002 &  0.004
    tpn               & -0.0777 &  -0.1089 &  0.0574 &  -0.0871 &  -0.0614 &  -0.0555
    vat               & -0.0018 &  0.0241 &  -0.0094 &  -0.0093 &  0.002 &  0.0011
    mean              -0.026075	-0.015975	0.03735	-0.0305	-0.02665	-0.01237

         
"""


labels = ['Ope', 'Con', 'Ext', "Agr", "Neu", "Avg"]
frame_ccc = [19.83, 17.03, 17.46, 13.96, 20.20, 17.70 ]  # audio
segment_ccc = [36.90,	40.93,	40.41,	28.87,	37.53,	36.93]
video_ccc = [38.60,	39.69,	38.77,	29.69,	39.24,	37.20]


x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(2 * x - 0.5, frame_ccc, width, label='Audio')
face = ax.bar(2 * x, segment_ccc, width, label='Visual')
face2 = ax.bar(2 * x + 0.5, video_ccc, width, label='Audiovisual')

x_line = np.arange(-1, 12, 2)
line = ax.plot(x_line, [0] * len(x_line), color="black", linewidth=1)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CCC (%)', fontsize=15)
# ax.spines['left'].set_position(("data", 0.5))
# ax.spines['bottom'].set_position(("data", 1))
# ax.set_title('Impression CCC scores by frame and face images')
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(labels)
ax.legend()


def autolabel_label(rects, xytxt=(-2, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        string = '{:.2f}'.format(height)
        if height < 0:
            height = height - 1.6
        ax.annotate(string,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytxt,  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
                    ha='center', va='bottom')


# def autolabel_face(rects, xytxt=(1, 2)):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=xytxt,  # 3 points vertical offset
#                     textcoords="offset points",
#                     fontsize=8,
#                     ha='center', va='bottom')


autolabel_label(frame, (-4, 1))
autolabel_label(face, (2, 6))
autolabel_label(face2, (10, 6))


fig.tight_layout()
plt.ylim(10, 50)
plt.savefig("impression_audio_visual_audiovisual_CCC.png", dpi=300)
plt.show()