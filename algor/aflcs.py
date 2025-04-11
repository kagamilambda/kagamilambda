import numpy as np
import torch
from loguru import logger

client_version = {}
client_num_samples = {}
client_update_times = {}
client_contribution_history = {}  # 跟踪客户端历史贡献质量
client_gradient_history = {}  # 当前模型和当前全局层的梯度余弦相似度


def aflcs(
        global_model,
        local_model,
        source=None,
        version=None,
        client_num_samples: dict = None,
        cfg=None,
        **kwargs,
):
    total_client = len(client_num_samples)
    client_weight = client_num_samples[source] / sum(client_num_samples.values())

    # 初始化客户端状态
    if client_version.get(source) is None:
        client_version[source] = 0
        client_update_times[source] = 0
        client_contribution_history[source] = []
        client_gradient_history[source] = {}
        for layer in global_model.keys():
            client_gradient_history[source][layer] = None

    # 计算基本陈旧度   version=update_received,每收到一个客户端更新就＋1，client_version[client id] 在最后会更新成version
    basic_staleness = version - client_version[source]

    # 1计算预期陈旧度 (考虑相对更新频率)
    relative_update_frequency = (version / total_client - client_update_times[source]) / cfg.num_rounds

    # ----------------------------------------------cfg.staleness_decay_rate是新加的，没有这个配置---------------------

    # 2计算时间衰减因子 - 指数衰减而不是线性 默认衰减率5是一个合理的起点，但可以考虑使其自适应，根据全局模型变化速率动态
    # 5次基础陈旧度是0.6，1次是0.9，10次是0.3
    time_decay_factor = np.exp(-basic_staleness / cfg.staleness_decay_rate) if hasattr(cfg,
                                                                                       'staleness_decay_rate') else np.exp(
        -basic_staleness / 10)

    # 计算梯度变化因子 (如果有历史梯度)
    gradient_change_factors = {}  # 当前层梯度差与之前的梯度之间的相似度
    combined_gradient_change = 0

    # 计算模型更新的总体贡献分数
    contribution_score = 0
    model_diff_magnitude = 0
    total_params = 0

    for layer in global_model.keys():
        # 3 梯度方向的一致性的计算。    持续提供相似方向梯度的客户端更可能贡献有价值的更新。
        if client_gradient_history[source][layer] is not None:
            current_gradient = local_model[layer].cpu() - global_model[layer].cpu()  # 当前梯度 = 模型层 - 全局层
            previous_gradient = client_gradient_history[source][layer]  # 之前的梯度 = 客户端梯度历史

            # 当前梯度与历史梯度的余弦相似度，范围为 [-1, 1]
            grad_sim = get_model_cosine_similarity(
                current_gradient.float().cpu().numpy().flatten(),
                previous_gradient.float().cpu().numpy().flatten()
            )
            '''
            for layer in global_model.keys():   #遍历模型的每一层
                cossim = get_model_cosine_similarity(   #计算当前层的全局模型和本地模型之间的余弦相似度
                    global_model[layer].float().cpu().numpy().flatten(),
                    local_model[layer].float().cpu().numpy().flatten(),
            )


            '''

            # 正梯度相似度表示客户端梯度方向稳定，应给予更高权重. 基础分0.5+ grad_sim 是当前梯度与历史梯度的余弦相似度，范围为 [-1, 1]
            gradient_consistency = 0.5 + (0.5 * grad_sim) if not np.isnan(
                grad_sim) else 0.5  # 如果是grad_sim NaN，则使用默认值 0.5
            gradient_change_factors[layer] = gradient_consistency
            combined_gradient_change += gradient_consistency  # 表示客户端梯度方向的整体一致性得分，是所有模型层的梯度一致性因子的平均值。

            # 更新梯度历史
            client_gradient_history[source][layer] = current_gradient.detach().clone()
        else:
            # 首次更新，初始化梯度历史
            client_gradient_history[source][layer] = (
                    local_model[layer].cpu() - global_model[layer].cpu()).detach().clone()
            gradient_change_factors[layer] = 0.5  # 先给一个中性值，其次再将梯度相似度赋值给他
            combined_gradient_change += 0.5

        # 计算更新幅度 (用于评估贡献质量)
        layer_diff = (local_model[layer].cpu() - global_model[
            layer].cpu()).abs()  # 表示特定层中客户端模型与全局模型之间的绝对差值 保留了更新的幅度信息而忽略方向 模型层梯度 - 全局层的梯度
        model_diff_magnitude += layer_diff.sum().item()  # 客户端模型与全局模型之间所有参数差值的累积和，代表总体更新幅度。
        # .sum() 计算张量中所有元素的总和。 .item将单元素张量转换为 Python 标量。
        total_params += layer_diff.numel()  # 计算的总参数数量，用于归一化更新幅度。.numel() 计算张量中元素的个数(NUMber of ELements)。

    # 归一化梯度变化因子  combined_gradient_change取值范围为0到1，值越高表示梯度方向越稳定
    if len(global_model.keys()) > 0:
        combined_gradient_change /= len(
            global_model.keys())  # 之前每一个layer combined_gradient_change都会相加，这里进行归一化，除以所有layer

    # 计算平均更新幅度 (用于评估异常值)
    avg_update_magnitude = model_diff_magnitude / max(1, total_params)

    # 更新客户端贡献历史
    client_contribution_history[source].append(avg_update_magnitude)  # 第source个客户端的贡献历史
    if len(client_contribution_history[source]) > 10:
        client_contribution_history[source].pop(0)  # 使用固定大小（10）的历史窗口是合理的，既能捕获趋势又不会过度依赖远古历史,pop0删除列表的第一个元素，并返回该元素的值。

    # 4计算历史贡献的一致性 (稳定的贡献应得到更高权重)
    # 变异系数（CV）是标准差与均值的比值，是评估相对离散程度的标准统计量，适合于不同尺度的比较。
    if len(client_contribution_history[source]) > 1:  # 检查特定客户端(source)是否有至少2条历史贡献记录
        history_mean = np.mean(client_contribution_history[source])  # 计算历史贡献的平均值:
        history_std = np.std(client_contribution_history[source])  # 计算历史贡献的标准差:
        cv = history_std / (history_mean + 1e-10)  # 变异系数   变异系数是标准化的离散程度度量，消除了量纲影响 加入1e-10避免除零错误
        contribution_stability = 1 / (1 + cv)  # 变异系数越小，稳定性越高，
    else:
        contribution_stability = 0.5  # 默认中等稳定性

    # 3自适应陈旧度补偿因子计算
    # 结合多种因素：基本陈旧度、梯度一致性、贡献稳定性和时间衰减
    staleness_compensation = (
            time_decay_factor *
            (0.4 + 0.3 * combined_gradient_change + 0.3 * contribution_stability)
        # 基础值+0.3*梯度一致性+0.3*贡献稳定性
    )

    # 这是补偿
    if relative_update_frequency >= 1:
        # 长期未更新的客户端，使用更谨慎的权重
        comp_weight = staleness_compensation / relative_update_frequency  # 刚刚计算出的补偿除以预期陈旧度
    elif relative_update_frequency >= 0:
        # 正常更新频率的客户端，适当奖励
        comp_weight = staleness_compensation * (1 + relative_update_frequency)
    else:
        comp_weight = 0

    # 应用改进的权重计算到每一层
    for layer in global_model.keys():
        cossim = get_model_cosine_similarity(
            global_model[layer].float().cpu().numpy().flatten(),
            local_model[layer].float().cpu().numpy().flatten(),
        )
        # 是当前模型层和全局模型层的余弦相似度。
        # 基于客户端数据量比例的基础权重
        alpha1 = client_weight

        # 基于余弦相似度的权重
        similarity_factor = (1 + cossim ** 2 * np.sign(cossim))  # 这里similarity_factor的范围是【0，2】
        if np.isnan(similarity_factor):
            similarity_factor = 0

        # 层特定梯度一致性调整
        layer_gradient_factor = gradient_change_factors.get(layer, 0.5)  # 获取当前层的梯度一致性因子，如果不存在则使用默认值0.5

        # 考虑层特定梯度变化的相似度权重
        alpha2 = similarity_factor * alpha1 * (0.5 + 0.5 * layer_gradient_factor)

        # 自适应beta - 根据陈旧度调整数据量权重与相似度权重的平衡
        # 随着陈旧度增加，相似度权重的重要性增加
        adaptive_beta = 0.3 * staleness_compensation

        # 归一化计算：结合数据量权重和相似度因子，并应用补偿
        alpha = (adaptive_beta * alpha1 + (1 - adaptive_beta) * alpha2) * (1 + comp_weight)

        # 防止单个客户端影响过大
        alpha = min(alpha, 0.8)

        # 应用陈旧度惩罚因子 - 对非常陈旧的更新进一步降低权重.如果客户端的陈旧度超过12，则对聚合权重alpha施加额外的惩罚。最多将alpha削减一半。
        if basic_staleness > 12:
            alpha = alpha * (1 - 0.05 * min(basic_staleness - 10, 10))

        alpha = torch.tensor(alpha).to(global_model[layer].device)

        logger.info(f"aggregate alpha: {alpha} (cossim: {cossim}, client_weight: {client_weight}, "
                    f"staleness: {basic_staleness}, comp_weight: {comp_weight}) layer: {layer}")

        global_model[layer] = (1 - alpha) * global_model[layer] + alpha * local_model[layer].cpu()

    # 更新客户端版本和更新次数
    client_version[source] = version
    client_update_times[source] += 1

    return global_model


def get_model_cosine_similarity(layer1, layer2):
    if np.all(layer1 == 0) or np.all(layer2 == 0):
        return 0
    return np.dot(layer1, layer2) / (np.linalg.norm(layer1) * np.linalg.norm(layer2) + 1e-10)


def advanced_staleness_metrics(
        global_models_history,
        client_update_history,
        source=None,
        version=None,
        cfg=None
):
    """
    计算高级陈旧度指标，可以作为辅助函数调用

    Parameters:
    -----------
    global_models_history : dict
        全局模型的历史快照，键为版本号
    client_update_history : dict
        客户端更新的历史记录，包含时间戳和贡献度
    source : str or int
        当前客户端标识
    version : int
        当前全局模型版本
    cfg : object
        配置对象

    Returns:
    --------
    dict
        包含多种陈旧度指标的字典
    """
    metrics = {}

    # 基础时间陈旧度
    if source in client_version:
        metrics['basic_staleness'] = version - client_version[source]
    else:
        metrics['basic_staleness'] = version

    # 相对更新频率
    if source in client_update_times:
        total_client = len(client_num_samples)
        metrics['update_frequency'] = client_update_times[source] / (version + 1)
        metrics['relative_frequency'] = (metrics['update_frequency'] * total_client)
    else:
        metrics['update_frequency'] = 0
        metrics['relative_frequency'] = 0

    # 贡献稳定性
    if source in client_contribution_history and len(client_contribution_history[source]) > 1:
        history = client_contribution_history[source]
        metrics['contribution_mean'] = np.mean(history)
        metrics['contribution_std'] = np.std(history)
        metrics['contribution_cv'] = metrics['contribution_std'] / (metrics['contribution_mean'] + 1e-10)
        metrics['contribution_stability'] = 1 / (1 + metrics['contribution_cv'])
    else:
        metrics['contribution_stability'] = 0.5

    # 全局变化速率 - 评估全局模型变化的速度
    if len(global_models_history) > 1:
        versions = sorted(global_models_history.keys())
        latest_versions = versions[-min(5, len(versions)):]

        changes = []
        for i in range(1, len(latest_versions)):
            prev_model = global_models_history[latest_versions[i - 1]]
            curr_model = global_models_history[latest_versions[i]]

            # 计算模型变化率
            total_change = 0
            total_params = 0

            for layer in prev_model.keys():
                if layer in curr_model:
                    diff = (curr_model[layer] - prev_model[layer]).abs()
                    total_change += diff.sum().item()
                    total_params += diff.numel()

            if total_params > 0:
                changes.append(total_change / total_params)

        if changes:
            metrics['global_change_rate'] = np.mean(changes)

            # 根据全局变化率调整陈旧度评估
            # 如果全局变化率高，那么陈旧度的影响更大
            metrics['adjusted_staleness'] = metrics['basic_staleness'] * (1 + metrics['global_change_rate'])
        else:
            metrics['global_change_rate'] = 0
            metrics['adjusted_staleness'] = metrics['basic_staleness']
    else:
        metrics['global_change_rate'] = 0
        metrics['adjusted_staleness'] = metrics['basic_staleness']

    return metrics