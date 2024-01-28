from parameters import InterpolatedPathPatch
from path_patching import interpolated_path_patch
import torch
from load_run import model, get_next_data
import glob

versions = glob.glob("saved_models/version_*")
versions = [int(v.split("_")[-1]) for v in versions]
if len(versions) == 0:
    version = 0
else:
    version = max(versions) + 1

LOAD=False
if LOAD:
    version = InterpolatedPathPatch.get_latest_version()
    patcher = InterpolatedPathPatch.load_by_version(version, model)
else:
    patcher = InterpolatedPathPatch(model, p_dropin=0.04, p_dropout=0.0)
optim = torch.optim.SGD(patcher.parameters(), lr = 0.3, momentum=0.2, nesterov=True)
optim = torch.optim.Adam(patcher.parameters(), lr=0.01, betas=(0.99, 0.997))
N = 1
l0_history = [1.]
import tqdm
# interpolated_path_patch = TimedFunc(interpolated_path_patch, print_on_call=True)
for i in tqdm.tqdm(range(10000)):
    print("\n\nstep ", i)
    optim.zero_grad()
    ioi_cache, abc_cache, ioi_metric, kl_metric = get_next_data(N)

    values, coefficients, logits = interpolated_path_patch(
        model, 
        coeffs=patcher,
        new_cache=abc_cache,
        orig_cache=ioi_cache,
        parallelism=12,
    )
    # prof.export_chrome_trace(f"./traces/traceall{i}.json")

    kl_div = kl_metric(logits) # this might not work with dropout/in
    loss_l1 = patcher.l1(0.001, 0.002, 0.001, crab = 0.)
    loss_logits = ioi_metric(logits)
    l0 = patcher.l0().item()
    l0_history.append(l0)
    # loss = kl_div * 0.01 + loss_logits
    loss = loss_logits
    loss = loss + loss_l1 + patcher.l_one_half(0.05)
    if i % 100 < 10 or i % 10 == 0:
        if l0 < 400:
            patcher.print_connections(threshold=0)
            patcher.print_connections()
        
        thresh = 0.5 * ((i // 10) % 5) / 5
        print("thresh", thresh)
        patcher.print_tp_fp(threshold=thresh)
        # patcher.print_tp_fp(threshold=0.5)

    print("intermeidate values:", patcher.num_intermediate())
    print("\nlogit diff:", loss_logits.item())
    print("nonzero coeffs:", l0)
    print("coeffs > 0.01:", patcher.l0(0.01).item())
    print("kl_score", kl_div.item())
    print("L1:", loss_l1.item())
    loss.backward()
    optim.step()
    if i <= 300 and i % 100 == 0:
        patcher.clamp_params(0.02)
    if i < 200 and i % 50 == 5:
        patcher.clamp_params(0.01)
        # patcher.reset_RS_coeffs(0.9, p=0.5, out=True)
        patcher.reset_edges(0.2, p=0.1)
    # if i == 10:
    #     patcher.clamp_params()
    # clamped_coefficients = patcher.clamplist()
    if i % 100 == 99:
        patcher.save(version = version, i = i)

