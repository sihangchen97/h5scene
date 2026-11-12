# h5scene metadata 键值决策(2026-11-13)

> **2026-11-13 补充决定(4 键定位键)**:来源三键 `scene_path` / `source_dataset` / `scene_ids`
> 退役,改为 4 个定位键 **`dataset` / `scene` / `seq` / `suffix`**。
> 语义(用户定):`dataset` = 顶层数据集名;`(dataset, scene[, seq])` 相同的多个 h5 =
> **同一 scene 的多视角相机**——在时序一致或 scene 静止时可拼成 N>1 的多相机数据集;
> `suffix` 标 scene 内的变体(train/val、相机/版本)。
> 依据:vggt_research/dataset 全量扫描发现目录深度 1~3 层不一(7scenes `<scene>/seq-NN.h5`、
> co3d `<object>/<instance>/{train,val}.h5`、artemis `<res>/<cls>/<id>.h5`、matterport 顶层即场景 id),
> 无机械规则能无损拆成固定 scene/seq 两层;且 vggt 侧 h5 本无 SpatialBench 三键(那是
> SpatialBench pipeline 独有)。4 键全可选、不校验(填错不影响,`meta_json` 原文兜底)。
> `split` / `tags` / `meta_json` / `frame_indices` 维持不变。

参考文件(旧 sparse_h5 格式,只读了结构与标量,未动数据):
- A: `SpatialBench/dataset/sparse_h5/7scenes/fire_seq-01_0.h5`(5 帧,3 MB,带 /spatialbench)
- B: `vggt_research/dataset/7scenes/chess/seq-05.h5`(1000 帧,510 MB,**无** /spatialbench)

## 1. 两个参考文件里实际存在什么

### file attrs
| key | A (fire, 5帧) | B (chess, 1000帧) | 备注 |
|---|---|---|---|
| `H` / `W` | 480 / 640 | 480 / 640 | 与我们的 `attrs.TNHW` 同位,但只有 HW、缺 T N |
| `nframes` | 5 | 1000 | 与 len(/rgb) 重复 |
| `has_metadata_source` | True | True | 含义不明,值恒 True(至少这两份) |

### /meta group
| key | 类型 | A | B | 观察 |
|---|---|---|---|---|
| `group_type` | str | `'temporal'` | `'temporal'` | 目前只见一种值 |
| `is_metric` | bool | True | True | 我们已存(/meta/is_metric) |
| `is_video` | bool | True | **缺** | B 没这个键 → 旧格式键集不稳定,读端必须按可选处理(我们也是) |
| `has_mask` / `has_pose` | bool | F / T | F / T | 可由 section 有无**派生**,冗余 |
| `images` | (T,) str | `'000112','000125',...` | `'000000','000001',...` | = /rgb 的 key(旧格式按源帧 id 命名帧!);我们 t_n 派生 key 后,源帧 id 只在 meta_json.frame_indices 里 |
| `max_interval` | int | 0 | **1000000** | B 的值明显是哨兵/垃圾(1000 连续帧,间隔应为 1) |
| `split` | (T,) str | 全 `'test'` | 全 `'test'` | 每帧重复一个文件级常量,浪费;应降为标量 |
| `timestamps` | (T,) f64 | 0.,1.,2.,3. | 0.,1.,2.,3. | 目前 = 重编号序号;A 是从视频抽帧(frame_indices 112,125,438,882,905),原始时间戳若有价值应存源秒数 |

### /spatialbench group(仅 A;B 完全没有)
| key | 值 | 备注 |
|---|---|---|
| `meta_json` | `{"depth_format": "uint16_png_div1000", "depth_unit": "meters", "frame_indices": [112,125,438,882,905], "intrinsic_mode": "shared", "n_frames": 5, "scene_ids": ["7scenes_fire_seq-01_0_sparse"], "scene_path": "fire_seq-01/0", "source_dataset": "7scenes", "tags": {...}}` | 来源的完整 JSON 原文;A 的 scene_path/scene_ids/n_frames 另有平铺拷贝 |
| `n_frames` | 5 | 与 file attr nframes 重复 |
| `scene_ids` | `'7scenes_fire_seq-01_0_sparse'` | 数据集侧的样本 id |
| `scene_path` | `'fire_seq-01/0'` | 相对源数据集的路径 |
| `source_dataset` | `'7scenes'` | 上游数据集名 |
| `tags/{data_type,dynamics,environment,view_density,view_type}` | `'real','static','indoor','sparse','normal'` | 5 个数据集侧标签,值域稳定 |

### 数据 section
- `/rgb/{frames}`:A/B 都是 **PNG 字节**(头 `\x89PNG`)→ 我们的 jpg95 策略会重编码。
- `/depth` (T,H,W) uint16,gzip,无 shuffle;采样均值 1710/1147 → **mm**(来源 meta_json 也确认 div1000)。
- `/K` (T,3,3)、`/T` (T,4,4) raw —— 与我们的 (T,N,·,·) 一致(N=1 退化)。
- `/groups`、`/group_scores`:变长 list-of-lists 的 data/lens/offsets 三件套(旧格式特有,内容未知,疑似逐帧候选点组)。

## 2. 我们 h5scene 现状(基线)

现状(2026-11-13 落地后):`attrs.TNHW` + `attrs.is_metric`(数据侧);`/rgb` jpg q≤95、
`/depth` uint16 mm、`/mask`、`/K`、`/T`、`/depth_conf`;`/meta/{dataset,scene,seq,suffix,
split,meta_json}` + `/meta/frame_indices` + `/meta/tags/`(数据集侧;4 键定位键见顶部补充决定)。
`set_meta(dataset, scene, seq, suffix, split, meta_json, frame_indices, tags, is_metric)`。
`is_video` 已废弃(T>1 即视频序列,轴本身表达,见 D7 后的补充决定)。

## 3. 决策清单(逐个定,✅=定了 ⬜=待定)

### 3.1 结构层
- ✅ **D1** 按语义分:与**数据本身**直接相关的进 file attrs(TNHW 类),与**数据集来源**相关的进 `/meta` group。
  由此:`is_metric` 从 /meta **上移 attrs**(数据量纲属性);`/meta` 收敛为"数据集上下文"。
- ✅ **D2** `meta_json` 原文**保留**,展开为结构化 key + 原文并存(无损;库不解析原文,只在 set_meta 时原样透传)。
- ✅ **D3** ~~三键平铺~~ → 被顶部 **4 键定位键决定**取代:改存 `dataset` / `scene` / `seq` / `suffix`。

### 3.2 帧级(轴 T)
- ✅ **D4** `timestamps` 源时间:暂不存,后面有需要再说(读端缺省 arange)。
- ✅ **D5** 见 D18(最终收为可选 `frame_indices`)。
- ✅ **D6** 源帧 id 字符串:不单独存,留在 meta_json 原文。

### 3.3 文件级
- ✅ **D7** `split`:存 **/meta/split 标量字符串**(数据集侧,按 D1);不存 (T,) 逐帧版。
- ⬜ **D8** `group_type`('temporal'/'...'):两份文件都是 'temporal',疑似预留枚举。
  建议:暂不存(我们的 T/N 轴已经表达了 temporal/multi-view);将来出现第二种值再收。
- ⬜ **D9** `max_interval`:B=1000000 是哨兵垃圾,A=0 没意义。建议:不存(派生自 timestamps/frame_indices)。
- ⬜ **D10** `has_mask`/`has_pose`:旧格式冗余位图。建议:不存(我们有 `has()` 从 section 存在性派生)。
- ⬜ **D11** `has_metadata_source`:含义不明且恒 True。建议:不存。
- ⬜ **D12** `nframes` 单独 attr:我们的 TNHW 已含 T。建议:不存。
- ✅ **D13** 标签:收,进 **/meta/tags/ 子组**,保留 tags 前缀,以后加标签不动结构(缺失的标签不写键)。
  实测值域(sparse_h5 全 129 个 h5,129/129 带 tags):
  | key | 取值(计数) |
  |---|---|
  | `data_type` | real(83) / simulation(46) |
  | `dynamics` | static(57) / dynamic(72) |
  | `environment` | indoor(103) / outdoor(26) |
  | `view_density` | sparse(129) —— 该库全是 sparse,无 dense 样本 |
  | `view_type` | normal(81) / wrist(42) / egoview(6) |
  `source_dataset` 实测 17 种:7scenes/adt/droid/dtu/eth3d/hiroom/nrgbd/omniworld/rlbench/robolab/
  robotwin/ropedia/scannetpp/tanks_and_temples/tum/vkitti/waymo。
  `scene_ids` 实测格式 = `{source_dataset}_{scene_path 斜杠转下划线}_sparse`(尾缀标处理链版本)。

### 3.4 来源与量纲
- ✅ **D14** key 名:由 **4 键定位键**决定 —— `dataset` / `scene` / `seq` / `suffix`(取代 `source_dataset`/`scene_path`/`scene_ids`)。
- ⬜ **D15** 深度量纲:我们固定 uint16 mm(0 与 65535 双哨兵)。旧文件与之一致(div1000)。
  建议:保持固定,不加 `depth_unit` key(值域写死,加 key 只留不确定的后门)。
- ⬜ **D16** `intrinsic_mode`('shared' = 全帧同一 K):我们是 (T,N,3,3) 广播存储,shared 天然表达。建议:不存。

### 3.5 旧格式特有,我们是否需要
- ✅ **D17** `/groups` + `/group_scores` — 已查实为**空壳占位**:两份文件都是
  单段 `lens=[T]`、data=`arange(T)`(0..T-1 恒等)、group_scores 全 1.0。
  不是逐帧变长列表,无任何信息 → **不收**,迁移时丢弃。
- ✅ **D18** 源帧回查:存**可选 /meta/frame_indices (T,) i64** —— 写入方提供且 ≠ arange(T) 时才写
  (dense 连续视频 t 本身就是源帧号,天然可查;稀疏抽帧如 A 的 112,125,438,882,905 才需要)。

## 4. 目标 schema(已定稿)

```
<scene>.h5
  # ---- attrs:数据侧(与数据本身直接相关)----
  TNHW          int64 (T, N, H, W)        (已有)
  is_metric     bool                        (从 /meta 上移;深度/尺度量纲属性)
  # ---- sections:数据 ----
  /rgb/{t}_{n}  uint8 1-D jpg q≤95 字节    (已有)
  /depth        uint16 (T,N,h,w) mm, gzip4+shuffle       (已有,h≤H w≤W 同比例)
  /mask         bool   (T,N,H,W), gzip4                     (已有)
  /K            f32 (T,N,3,3) raw                           (已有)
  /T            f32 (T,N,4,4) raw                           (已有)
  /depth_conf   f16 (T,N,h,w) 0-1, gzip4, 与 depth 同 (h,w) (已有)
  # ---- /meta:数据集侧(来源/上下文)----
  /meta/dataset                  str             (顶层数据集名;定位键 1,可选)
  /meta/scene                    str             (场景;定位键 2 —— (dataset,scene[,seq])
                                                 相同的多 h5 = 该 scene 多视角相机,可选)
  /meta/seq                      str             (序列;定位键 3,可选)
  /meta/suffix                   str             (变体:train/val、相机/版本;定位键 4,可选)
  /meta/split                    str             (标量,可选;train/val/test)
  /meta/meta_json                str             (来源 JSON 原文,可选,无损透传)
  /meta/frame_indices            int64 (T,)     (可选,仅 ≠ arange(T) 时写;稀疏抽帧回查)
  /meta/tags/{data_type,dynamics,environment,view_density,view_type}   str (各可选)
```

### 迁移规则(旧 sparse_h5 → 新格式)
- `/rgb` PNG 字节 → 按 q≤95 策略重编码;旧 key(源帧号)→ 新 t_n 派生 key,源帧号进 `frame_indices`。
- `/depth` 全分辨率 mm → 原样可存(也可下采样存小);`/K` `/T` 形状直接兼容(N=1)。
- `/meta` 旧键中:`is_metric` 上移 attrs;`split`/`tags` 收入新 /meta;
  `is_video` 与 `images`/`max_interval`/`has_mask`/`has_pose`/`group_type`/`timestamps`/
  `has_metadata_source`/`nframes` 一并**丢弃**(T/N 轴已表达视频性);
  `spatialbench/meta_json` 原文 → /meta/meta_json;4 键按目录名填:
  `dataset` = 顶层目录,`scene`/`seq`/`suffix` 由 scene_path 按数据集约定拆
  (如 7scenes `fire_seq-01/0` → scene=fire, seq=seq-01, suffix=0)。
- `/groups` `/group_scores` 空壳占位(查实为 arange(T) + 全 1.0)→ **丢弃**。
