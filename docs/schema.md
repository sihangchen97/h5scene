# h5scene 文件 schema

> 以代码为准:`h5scene/h5base.py`(句柄 + 通用访问器)、`h5scene/h5cam.py`、
> `h5scene/h5camrig.py`、`h5scene/codecs/sections.py`(section 布局表:
> dtype / 压缩 / shuffle / fillvalue)。改格式时同步更新本文档、`README.md`、
> `h5scene/__init__.py` 的 docstring。

## H5Base(`h5base.py`)

纯文件句柄,无形状、无已知键。自带 `ATTR_KEYS` / `META_KEYS` 均为空,
`attr()`/`set_attr()`/`meta()`/`set_meta()` 都要求名字在子类注册表里。

文件级约定(create 时):
- 根 attrs 空(轴属性由子类 pin);
- `/meta` 组(create 时自动建,可空);
- 通用写入器 `write_section(name, data, **kw)`(先删后建,替换语义)。

`set_meta` 两种等价形态:单键 `set_meta(name, value)`(None = 删除)与多键
`set_meta(dataset=…, …)`(None = 不动)。值先校验、后删旧值。

## H5Cam(`h5cam.py`)— 单相机 (T, H, W)

```
<cam>.h5
  # ---- 根 attrs(数据侧)----
  file_type   str = "h5cam"               必有(create pin;类型标签,构造器拒不符)
  THW         int64 (3,) = (T, H, W)        必有
  is_metric   bool                        必有(构造默认 False)
  # ---- sections ----
  /img/{t}    uint8 1-D 图像字节 jpg q<=95   可空组;有则 key 必须 0..T-1 完整
  /depth      uint16 (T, h, w) mm;0 与 65535 双哨兵;gzip4 + shuffle;
              h<=H w<=W 且 H/W 比例一致;读回 resize 回 (H,W)          [可选]
  /mask       bool (T, H, W);1 = valid,gzip4;整帧 padding 可 (T,) 广播  [可选]
  /K          f32 (T, 3,3) raw;(3,3) 输入广播;约定 = OpenCV pinhole:
              fx K[0,0] / fy K[1,1] > 0,K[2,2] = 1,skew K[1,0] = 0,
              K[2,1] = K[2,0] = 0;set_K 对每一帧校验(check_K)      [可选]
  /T          f32 (T, 4,4) c2w raw;(4,4) 输入广播;约定 = camera-to-world:
              末行 [0,0,0,1],R = T[:3,:3] 正交且 det = 1;
              set_pose_c2w 对每一帧校验(check_pose)               [可选]
              # 像素约定:主点 (u, v) = K[0,2] / K[1,2];像素 (u, v) 即点 (u, v)
  /depth_conf f16 (T, h, w) 0-1,gzip4,与 /depth 同 (h, w)           [可选]
              # 输入只收 0-1 float(set_depth_conf_01,整型拒绝)
  # ---- /meta(数据集侧,全部可选;缺失不写键)----
  /meta/dataset        str 顶层数据集名(定位键 1)
  /meta/scene          str 场景(定位键 2)
  /meta/seq            str 序列(定位键 3,单相机才有)
  /meta/suffix         str 变体标记(train/val、相机/版本;定位键 4)
  /meta/split          str 标量(train/val/test)
  /meta/src_frame_idxs  int64 (T,) 源帧号;仅 != arange(T) 时才写(dense 丢弃)
  /meta/src_frame_ids   utf-8 (T,) 源帧 ID(字符串:tum 时间戳 / dl3dv 路径);
                        必须唯一;rig 无此键(合并轴无单一意义)
  /meta/timestamps      f64 (T,) 源帧时间戳(整型输入升 f64)
  /meta/tags/*         str 数据集侧标签(data_type/dynamics/environment/
                        view_density/view_type …),/meta/tags/ 子组,缺失的标签不写键
```

- 定位语义:`(dataset, scene[, seq])` 相同的多个 h5 = 同一 scene 的多视角相机,
  可用 `H5CamRig` 合并;`suffix` 标变体。
- open 时校验:`THW` 必有;img key 0..T-1 完整;各 section 轴 0 = T、
  K/T 帧形 (3,3)/(4,4)、mask 全尺寸 (H,W)、depth/depth_conf 比例一致
  且两者 (h,w) 相同。
- `depth_mask(t, conf_min=…)` 为派生(不存储)。

## H5CamRig(`h5camrig.py`)— 多相机 (T, N, Hmax, Wmax),零数据纯引用

```
<rig>.h5
  # ---- 根 attrs ----
  file_type  str = "h5camrig"               必有(build pin;类型标签,构造器拒不符)
  TNHW       int64 (4,) = (T, N, Hmax, Wmax)  必有
  is_metric  bool                               必有(继承自相机,须一致)
  cam_files  str[]  源相机文件路径(相对 rig 文件所在目录;VDS/ExternalLink
                    同按该目录解析 → rig 与相机一起拷贝/移动后仍可打开)  必有(N 个)
  cam_hw     int64 (N, 2) 每相机原生 (H, W)     必有
  (cam_hw 即相机的 (H, W);open 时从相机文件重读 attrs.THW,须等于
                    (T, *cam_hw[n]),且校验每个被映射 section 的存在与形状)  必有
  # ---- sections(全部零字节;img 走链接,其余走 VDS)----
  /img/{t}_{n}   ExternalLink → 相机 n 的 /img/{t}(jpg 变长不能 VDS);
                 rgb(t, n) 按相机 n 原生 (H, W) 解码
  /depth         VDS uint16 (T, N, Hmax, Wmax),fillvalue 0
  /mask          VDS bool   (T, N, Hmax, Wmax),fillvalue False
  /K             VDS f32    (T, N, 3, 3),每相机一整块映射
  /T             VDS f32    (T, N, 4, 4),每相机一整块映射
  /depth_conf    VDS f16    (T, N, Hmax, Wmax),fillvalue 0.0
  # 小相机逐帧 block 映射 (t, n, :h, :w),余区由 fillvalue 覆盖(0/False 恰是存储哨兵)
  # ---- /meta(从相机继承;只存非空值)----
  /meta/dataset      str 相机们必须一致(build 时 check)
  /meta/scene        str 相机们必须一致(build 时 check)
  /meta/suffix       str rig 自己的变体标记(build 参数,默认空 = 不写)
  /meta/split        str 所有相机一致才继承
  /meta/tags/*       str 所有相机一致才继承(/meta/tags/ 子组)
  # 没有 seq(合并后就是时间轴本身),也没有 src_frame_idxs(各源帧号对合并轴无单一意义)
```

- build 校验(全在写文件前;输出目录自动建,write-temp-then-rename,
  失败不碰已有输出):T 一致、is_metric 一致、dataset 一致、scene 一致、
  img 全有或全无、section 集合一致、img 每帧尺寸 == 声明 (H,W)、
  depth/depth_conf 必须存相机原生 (H,W)(下采样深度拒绝,VDS 不能上采样)、
  H 或 W 全相机相同(另一边取 max,左上对齐;两边都不同拒绝)、
  **输出路径不得是任一相机**(否则会覆盖相机数据)。
- open 校验:`cam_files` 存在,且相机 attrs.THW(从相机文件重读)仍等于
  `(T, *cam_hw[n])`、每个被映射 section 仍在且形状未变 ——源文件被删/改写
  THW 时 VDS 会静默读 fillvalue,必须显式报错。不查 size/mtime:cp/rsync
  或相机文件的 meta 编辑不影响打开;改 THW、删 section、改 section 形状才
  必须报错。
