

import subprocess
from collections import defaultdict


EXCLUDE_LIST = []  # exclude GPUs from these labs from the count


def _get_valid_node_info(parse_only_gpu_nodes=True):
    """Call sinfo and parse into a structure of dicts.

    node_name:
        field_name: value
        ...
    ...
    """
    # sinfo can be queried with --Format options:
    # We split into two queries to get all info into separate rows
    query_health = "NodeList:22,Available:6,Reason:40,StateComplete:20"
    query_layout = "NodeList:22,Features:40,Gres:35,GresUsed:54,Memory:10,AllocMem:10,CPUsState:10"

    sinfo_state_health = str(subprocess.check_output(["sinfo", "--Node", "--Format", query_health])).split("\\n")
    field_names = sinfo_state_health[0].split()
    assert field_names[1:] == ["AVAIL", "REASON", "STATECOMPLETE"]
    node_info = dict()
    for entry in sinfo_state_health[2:-1]:
        # Use manual offsets here as defined above.
        # This is necessary because the REASON field may contain whitespaces.
        row_info = [field.strip() for field in [entry[0:22], entry[22:28], entry[28:68], entry[68:]]]
        node_info[row_info[0]] = dict(zip(field_names[1:], row_info[1:]))

    sinfo_state_layout = str(subprocess.check_output(["sinfo", "--Node", "--Format", query_layout])).split("\\n")
    field_names = sinfo_state_layout[0].split()
    field_names[-1] = "CPUs"
    assert field_names[1:] == ["AVAIL_FEATURES", "GRES", "GRES_USED", "MEMORY", "ALLOCMEM", "CPUs"]
    for entry in sinfo_state_layout[2:-1]:
        # Here we can simply split along whitespaces
        row_info = entry.split()
        node_info[row_info[0]].update(dict(zip(field_names[1:], row_info[1:])))

    if parse_only_gpu_nodes:
        node_info = {k: v for k, v in node_info.items() if v["GRES"] != "(null)"}
    return node_info


def _check_gpus():
    total_resources = dict(gpus=0, cpus=0, mem=0)
    overall_free = dict(gpus=0, cpus=0, mem=0, gpus_not_starved=0)
    gpu_free_by_type = defaultdict(int)
    gpu_not_starved_by_type = defaultdict(int)
    print("Printing free resources on each node:")
    node_dict_query = _get_valid_node_info()
    for node, node_info in node_dict_query.items():
        if not any([exclude in node for exclude in EXCLUDE_LIST]):
            try:
                node_spec = f"[{node_info['AVAIL_FEATURES']},{node_info['GRES']},{int(node_info['MEMORY'])//1024} GB Mem]: "
                total_resources["mem"] += int(node_info["MEMORY"])
                total_resources["cpus"] += int(node_info["CPUs"].split("/")[3])  # allocated/idle/other/total
                if node_info["GRES"] != "(null)":
                    # Parse GRES groups that look like gpu:rtx2080ti:0(IDX:N/A),gpu:rtxa4000:1(IDX:7)
                    total_resources["gpus"] += sum(
                        [int(gres_total.split("(")[0].split(":")[-1]) for gres_total in node_info["GRES"].split(",gpu:")]
                    )

                if node_info["AVAIL"] != "up":
                    print(f"{node:10}: down. Reason: {node_info['REASON']}. State: {node_info['STATECOMPLETE']}    {node_spec}")
                    overall_free["mem"] += 0
                    overall_free["cpus"] += 0
                    overall_free["gpus"] += 0
                elif node_info["STATECOMPLETE"] not in ["idle", "mixed", "allocated"]:
                    print(f"{node:11}: unavailable. State: {node_info['STATECOMPLETE']}. Reason: {node_info['REASON']}. {node_spec}")
                    overall_free["mem"] += 0
                    overall_free["cpus"] += 0
                    overall_free["gpus"] += 0
                else:
                    memleft = int(node_info["MEMORY"]) - int(node_info["ALLOCMEM"])
                    cpuleft = int(node_info["CPUs"].split("/")[1])  # allocated/idle/other/total
                    gpuleft = 0
                    if node_info["GRES"] != "(null)":
                        # Parse GRES groups that look like gpu:rtx2080ti:0(IDX:N/A),gpu:rtxa4000:1(IDX:7)
                        for gres_total, gres_group in zip(node_info["GRES"].split(",gpu:"), node_info["GRES_USED"].split(",gpu:")):
                            gpu_avail = int(gres_total.split("(")[0].split(":")[-1])
                            gpu_alloc = int(gres_group.split("(")[0].split(":")[-1])
                            gpuleft += gpu_avail - gpu_alloc
                            gpu_free_by_type[gres_group.split("gpu:")[-1].split(":")[0]] += gpu_avail - gpu_alloc
                    print(f"{node:11}: {gpuleft:2} GPUs | {cpuleft:3} CPUs |  {memleft//1024:4} GB MEM |    {node_spec}")
                    overall_free["gpus"] += gpuleft
                    overall_free["mem"] += memleft
                    overall_free["cpus"] += cpuleft
                    overall_free["gpus_not_starved"] += min(min(cpuleft // 4, memleft // 1024 // 31), gpuleft)
                    if node_info["GRES"] != "(null)":
                        # Loop a 2nd time to roughly estimate starved GPUs by type.
                        assignable_gpus = min(min(cpuleft // 4, memleft // 1024 // 31), gpuleft)
                        for gres_total, gres_group in zip(node_info["GRES"].split(",gpu:"), node_info["GRES_USED"].split(",gpu:")):
                            if assignable_gpus > 0:
                                gpu_avail = int(gres_total.split("(")[0].split(":")[-1])
                                gpu_alloc = int(gres_group.split("(")[0].split(":")[-1])
                                assigned_to_group = min(assignable_gpus, gpu_avail - gpu_alloc)
                                gpu_not_starved_by_type[gres_group.split("gpu:")[-1].split(":")[0]] += assigned_to_group
                                assignable_gpus -= assigned_to_group
                            else:
                                gpu_not_starved_by_type[gres_group.split("gpu:")[-1].split(":")[0]] += 0

            except Exception as e:
                print(f"Script error for node {node}. Error: {e}")
        else:
            print(f"{node:11}: not included in count. Node not available to general users.")

    if overall_free["gpus"] > overall_free["gpus_not_starved"]:
        starvation_msg = f" (but only {overall_free['gpus_not_starved']} usable, others are starved for CPU/MEM resources)"
    else:
        starvation_msg = ""

    print("----------------------------------------------------------")
    print(
        f"Summary: GPUs: {overall_free['gpus']}/{total_resources['gpus']} card{'s' if overall_free['gpus'] > 1 else ''} available{starvation_msg}.\n"
        f"         CPUs: {overall_free['cpus']}/{total_resources['cpus']} chip{'s' if overall_free['cpus'] > 1 else ''} available.\n"
        f"         MEM : {overall_free['mem']//1024:0.0f}/{total_resources['mem']//1024:0.0f} GB available."
    )
    print("----------------------------------------------------------")

    if len(gpu_free_by_type) > 0:
        print("Availability by GPU type: ")
        entry_str_width = max([len(entry) for entry in gpu_free_by_type.keys()])
        for (entry1, val1), (entry2, val2) in zip(
            dict(sorted(gpu_free_by_type.items())).items(), dict(sorted(gpu_not_starved_by_type.items())).items()
        ):
            print(f"Type {entry1:<{entry_str_width}} - {val1:>3} card{'s' if val1 > 1 else ' '} available, {val2:>3} usable.")
    else:
        print("No GPUs found.")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    _check_gpus()