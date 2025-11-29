import scipy.stats as stats

# 模型准确率数据（独立样本）
unetpp_acc = [0.9664666929324767, 0.9666338344041702, 0.9664244150806157, 0.9665624621166202, 0.9666661110437629]
CE_acc = [0.9679283852588192, 0.967545914656322, 0.9678574675718268, 0.9677210874045338, 0.9676312280276397]
DU_acc = [0.9681255303673172, 0.9689313856224997, 0.9670440356406838, 0.9682988847132986, 0.9694978179173231]
PCRTAM_acc = [0.9704788459207178, 0.970699478724694, 0.9706100739483574, 0.9703242817311188, 0.9707746393502241]
GT_acc = [0.970017729421748, 0.9701766880834042, 0.969968935628561, 0.9697561825675839, 0.9698945326706268]
our_method_acc = [0.9712054491453509, 0.971236665050309, 0.971127712449994, 0.9712124196872349, 0.9713665292762759]

# 计算 our_method 与 U-Net++ 之间的 t 检验
t_stat_our_vs_unetpp, p_value_our_vs_unetpp = stats.ttest_ind(our_method_acc, unetpp_acc)

# 计算 our_method 与 AttU-Net 之间的 t 检验
t_stat_our_vs_CE_acc, p_value_our_vs_CE_acc = stats.ttest_ind(our_method_acc, CE_acc)

# 计算 our_method 与 LadderNet 之间的 t 检验
t_stat_our_vs_DU_acc, p_value_our_vs_DU_acc = stats.ttest_ind(our_method_acc, DU_acc)

# 计算 our_method 与 LadderNet 之间的 t 检验
t_stat_our_vs_PCRTAM_acc, p_value_our_vs_PCRTAM_acc = stats.ttest_ind(our_method_acc, PCRTAM_acc)


# 计算 our_method 与 LadderNet 之间的 t 检验
t_stat_our_vs_GT_acc, p_value_our_vs_GT_acc = stats.ttest_ind(our_method_acc, GT_acc)


# 输出结果
# 输出结果，保留更多的小数位
print(f"our_method vs unetpp: t-statistic = {t_stat_our_vs_unetpp:.4f}, p-value = {p_value_our_vs_unetpp:.15e}")
print(f"our_method vs CE_acc-Net: t-statistic = {t_stat_our_vs_CE_acc:.4f}, p-value = {p_value_our_vs_CE_acc:.15e}")
print(f"our_method vs DU_acc: t-statistic = {t_stat_our_vs_DU_acc:.4f}, p-value = {p_value_our_vs_DU_acc:.15e}")
print(f"our_method vs PCRTAM: t-statistic = {t_stat_our_vs_PCRTAM_acc:.4f}, p-value = {p_value_our_vs_PCRTAM_acc:.15e}")
print(f"our_method vs GT: t-statistic = {t_stat_our_vs_GT_acc:.4f}, p-value = {p_value_our_vs_GT_acc:.15e}")
