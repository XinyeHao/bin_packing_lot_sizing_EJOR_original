# ceshi.py
from preprocessor.preprocess import load_instance


def check_instance(name):
    print(f"\n========== 检查算例: {name} ==========")
    inst = load_instance(name)

    # 调试：打印所有属性名，方便了解对象结构
    print("可用属性:", [attr for attr in dir(inst) if not attr.startswith('_')])

    # ---- 获取物料列表 ----
    if hasattr(inst, 'item_ids'):
        item_ids = inst.item_ids
    elif hasattr(inst, 'items'):
        item_ids = inst.items
    elif hasattr(inst, 'item_config'):
        item_ids = list(inst.item_config.keys())
    else:
        raise AttributeError("无法找到物料ID列表属性，请查看上方打印的可用属性列表。")

    print(f"物料数 I: {len(item_ids)}")
    print(f"时段数 T: {inst.num_periods}")
    print(f"机台数 M: {len(inst.machine_ids)}")
    print(f"配置数 U: {len(inst.config_ids)}")

    # ---- 需求数据（支持两种结构） ----
    demand_raw = getattr(inst, 'demand', {})
    if not demand_raw:
        print("⚠️  未找到需求数据（demand 为空或不存在）。")
        total_demand = 0
        demand = {}
    else:
        # 检测结构：如果第一个值是字典，则嵌套；否则扁平
        first_val = next(iter(demand_raw.values())) if demand_raw else None
        if isinstance(first_val, dict):
            # 嵌套结构 {i: {t: d}}
            print("检测到嵌套需求结构 {物料: {时段: 需求量}}")
            # 展平为 {(i,t): d}
            demand = {}
            for i, t_dict in demand_raw.items():
                for t, d in t_dict.items():
                    demand[(i, t)] = d
        else:
            # 扁平结构 {(i,t): d}
            print("检测到扁平需求结构 {(物料,时段): 需求量}")
            demand = demand_raw

        total_demand = sum(demand.values()) if demand else 0
        print(f"需求总数: {total_demand}")

    if total_demand == 0:
        print("⚠️  需求总量为 0，这解释了为什么所有物料都报废（生产无意义）。")
    else:
        print("需求非零，示例需求条目（前10个）:")
        for idx, ((i, t), d) in enumerate(demand.items()):
            if idx >= 10:
                break
            print(f"  物料 {i}, 时段 {t}: {d}")

    # ---- 报废成本（使用 scrap_costs 属性） ----
    rc_dict = getattr(inst, 'scrap_costs', {})
    if rc_dict and item_ids:
        rc = rc_dict.get(item_ids[0], "未定义")
    else:
        rc = "未定义"
    print(f"示例报废成本 rc_{item_ids[0] if item_ids else '?'}: {rc}")
    if rc != "未定义" and rc < 100:
        print("   ⚠️  报废成本偏低（<100），可能使报废比生产更经济。")

    # ---- 每个物料的需求量 ----
    print("\n各物料需求情况（仅列出有需求的）:")
    item_demand_count = {i: 0 for i in item_ids}
    for (i, t), d in demand.items():
        if i in item_demand_count:
            item_demand_count[i] += d
    has_demand = False
    for i, cnt in item_demand_count.items():
        if cnt > 0:
            print(f"  物料 {i}: {cnt} 件需求")
            has_demand = True
    if not has_demand:
        print("  所有物料需求均为 0。")

    # ---- 建议 ----
    print("\n建议:")
    if total_demand == 0:
        print("- 修改算例生成逻辑，为物料添加非零需求（参考 Set-C 生成方式）。")
    if rc != "未定义" and rc < 100:
        print("- 考虑调高报废成本 rc_i（例如设为 500~1000），鼓励生产。")
    print("=" * 40)


if __name__ == "__main__":
    check_instance("set_a_01")