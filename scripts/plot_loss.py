path="output/cpm/resnet50-cpm-resnet-cropped-flipped_rotated-masked1578541074.4775078-train.log"
import matplotlib.pyplot as plt
with open(path, "rt") as f:
    losses =[]
    for l in f:
        try:
            idx0 = l.strip().index('pafmaps_stage_0=')+len("pafmaps_stage_0=")
            idx1 = l.strip().index(',batch_loss_heatmaps_stage_1')
            l = l.strip()[idx0:idx1]
            losses.append(float(l))
            print(l)
        except:
            pass
    plt.plot(losses)
    plt.show()