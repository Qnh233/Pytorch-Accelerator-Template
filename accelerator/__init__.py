from accelerator.CustomTracker import TensorboardCustomTracker


# 自定义组件注册
def get_tracker(name: str):
    """根据名称获取追踪器类"""
    tracker_registry = {
        "TensorboardCustomTracker": TensorboardCustomTracker,
    }

    if name not in tracker_registry:
        raise ValueError(f"Unknown tracker: {name}. Available: {list(tracker_registry.keys())}")

    return tracker_registry[name]


def create_tracker(tracker_config: dict):
    """根据配置创建追踪器实例"""
    if tracker_config is None:
        return None
    tracker_class = get_tracker(tracker_config["name"])
    params = tracker_config.get("params", {}) or {}  # 确保params不会是None
    return tracker_class(**params)

