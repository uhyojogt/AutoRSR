import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Type, Union


DOMAIN_NAMES = ("ningbo", "daqing", "japan")
DOMAIN_NAME_TO_ID = {name: idx for idx, name in enumerate(DOMAIN_NAMES)}
DOMAIN_ID_TO_NAME = {idx: name for name, idx in DOMAIN_NAME_TO_ID.items()}


def normalize_domain_name(name, default: str = "ningbo") -> str:
    if name is None:
        return default
    raw = str(name).strip()
    if not raw:
        return default
    lower = raw.lower()

    if lower in DOMAIN_NAME_TO_ID:
        return lower
    if "daqing" in lower or "大庆" in raw:
        return "daqing"
    if "japan" in lower or "日本" in raw:
        return "japan"
    if "ningbo" in lower or "宁波" in raw or "nb_" in lower or lower == "nb":
        return "ningbo"
    return default


def domain_name_to_id(name, default: str = "ningbo") -> int:
    return DOMAIN_NAME_TO_ID[normalize_domain_name(name, default=default)]


def guess_domain_from_text(text, default: str = "ningbo") -> str:
    return normalize_domain_name(text, default=default)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DomainSpecificBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_domains: int = len(DOMAIN_NAMES),
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_domains = int(num_domains)
        self.default_domain = 0
        self.bns = nn.ModuleList(
            [
                nn.BatchNorm2d(
                    num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                )
                for _ in range(self.num_domains)
            ]
        )

    def set_default_domain(self, domain) -> None:
        self.default_domain = int(domain_name_to_id(domain) if isinstance(domain, str) else domain)

    def _coerce_domain_ids(self, domain_ids, batch_size: int, device: torch.device) -> torch.Tensor:
        if domain_ids is None:
            return torch.full((batch_size,), self.default_domain, device=device, dtype=torch.long)
        if isinstance(domain_ids, str):
            return torch.full((batch_size,), domain_name_to_id(domain_ids), device=device, dtype=torch.long)
        if isinstance(domain_ids, int):
            return torch.full((batch_size,), int(domain_ids), device=device, dtype=torch.long)
        if not torch.is_tensor(domain_ids):
            domain_ids = torch.as_tensor(domain_ids, dtype=torch.long, device=device)
        else:
            domain_ids = domain_ids.to(device=device, dtype=torch.long)
        if domain_ids.ndim == 0:
            domain_ids = domain_ids.view(1).repeat(batch_size)
        if domain_ids.numel() == 1:
            domain_ids = domain_ids.view(1).repeat(batch_size)
        if domain_ids.numel() != batch_size:
            raise ValueError(
                f"Domain id count must match batch size. Got {domain_ids.numel()} ids for batch size {batch_size}."
            )
        return domain_ids.view(batch_size)

    def forward(self, x: Tensor, domain_ids=None) -> Tensor:
        domain_ids = self._coerce_domain_ids(domain_ids, batch_size=x.shape[0], device=x.device)
        if torch.all(domain_ids == domain_ids[0]):
            return self.bns[int(domain_ids[0].item())](x)

        out = torch.empty_like(x)
        for domain_id in torch.unique(domain_ids).tolist():
            mask = domain_ids == int(domain_id)
            out[mask] = self.bns[int(domain_id)](x[mask])
        return out


class DSBNDownsample(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, stride: int, num_domains: int) -> None:
        super().__init__()
        self.conv = conv1x1(inplanes, outplanes, stride)
        self.bn = DomainSpecificBatchNorm2d(outplanes, num_domains=num_domains)

    def forward(self, x: Tensor, domain_ids=None) -> Tensor:
        return self.bn(self.conv(x), domain_ids=domain_ids)


class DSBNBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        num_domains: int = len(DOMAIN_NAMES),
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in DSBNBasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains=num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains=num_domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, domain_ids=None) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, domain_ids=domain_ids)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, domain_ids=domain_ids)

        if self.downsample is not None:
            identity = self.downsample(x, domain_ids=domain_ids)

        out += identity
        out = self.relu(out)
        return out


class DSBNDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_domains: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = DomainSpecificBatchNorm2d(out_channels, num_domains=num_domains)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1: Tensor, x2: Tensor, domain_ids=None) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        x = self.bn(x, domain_ids=domain_ids)
        x = self.relu(x)
        return x


class DSBNDecodeHead(nn.Module):
    def __init__(self, num_classes: int, num_domains: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn = DomainSpecificBatchNorm2d(32, num_domains=num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.cls = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

    def forward(self, x: Tensor, domain_ids=None) -> Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, domain_ids=domain_ids)
        x = self.relu(x)
        x = self.cls(x)
        return x


class DSBNResNet(nn.Module):
    def __init__(
        self,
        block: Type[DSBNBasicBlock],
        layers,
        num_classes: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        num_domains: int = len(DOMAIN_NAMES),
    ) -> None:
        super().__init__()
        self.num_domains = int(num_domains)
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(self.inplanes, num_domains=self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_domains=self.num_domains)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            num_domains=self.num_domains,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            num_domains=self.num_domains,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            num_domains=self.num_domains,
        )

        channels = [64, 64, 128, 256, 512]
        self.decode4 = DSBNDecoder(channels[4], channels[3], num_domains=self.num_domains)
        self.decode3 = DSBNDecoder(channels[3], channels[2], num_domains=self.num_domains)
        self.decode2 = DSBNDecoder(channels[2], channels[1], num_domains=self.num_domains)
        self.decode1 = DSBNDecoder(channels[1], channels[0], num_domains=self.num_domains)
        self.decode0 = DSBNDecodeHead(num_classes=num_classes, num_domains=self.num_domains)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, DomainSpecificBatchNorm2d):
                for bn in m.bns:
                    nn.init.constant_(bn.weight, 1)
                    nn.init.constant_(bn.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DSBNBasicBlock):
                    for bn in m.bn2.bns:
                        nn.init.constant_(bn.weight, 0)

    def _make_layer(
        self,
        block: Type[DSBNBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        num_domains: int = len(DOMAIN_NAMES),
    ) -> nn.ModuleList:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DSBNDownsample(self.inplanes, planes * block.expansion, stride, num_domains=num_domains)

        layers = nn.ModuleList()
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                num_domains=num_domains,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    num_domains=num_domains,
                )
            )
        return layers

    @staticmethod
    def _forward_layer(layer: nn.ModuleList, x: Tensor, domain_ids=None) -> Tensor:
        for block in layer:
            x = block(x, domain_ids=domain_ids)
        return x

    def set_default_domain(self, domain) -> None:
        domain_id = domain_name_to_id(domain) if isinstance(domain, str) else int(domain)
        for module in self.modules():
            if isinstance(module, DomainSpecificBatchNorm2d):
                module.set_default_domain(domain_id)

    def _forward_impl(self, x: Tensor, domain_ids=None) -> Tensor:
        encoder = []

        x = self.conv1(x)
        x = self.bn1(x, domain_ids=domain_ids)
        x = self.relu(x)
        encoder.append(x)

        x = self.maxpool(x)
        x = self._forward_layer(self.layer1, x, domain_ids=domain_ids)
        encoder.append(x)

        x = self._forward_layer(self.layer2, x, domain_ids=domain_ids)
        encoder.append(x)

        x = self._forward_layer(self.layer3, x, domain_ids=domain_ids)
        encoder.append(x)

        x = self._forward_layer(self.layer4, x, domain_ids=domain_ids)
        encoder.append(x)

        d4 = self.decode4(encoder[4], encoder[3], domain_ids=domain_ids)
        d3 = self.decode3(d4, encoder[2], domain_ids=domain_ids)
        d2 = self.decode2(d3, encoder[1], domain_ids=domain_ids)
        d1 = self.decode1(d2, encoder[0], domain_ids=domain_ids)
        out = self.decode0(d1, domain_ids=domain_ids)
        return out

    def forward(self, x: Tensor, domain_ids=None) -> Tensor:
        return self._forward_impl(x, domain_ids=domain_ids)


def resnet34_dsbn(
    num_class: Optional[int] = None,
    num_domains: int = len(DOMAIN_NAMES),
    num_classes: Optional[int] = None,
) -> DSBNResNet:
    if num_class is None:
        num_class = num_classes
    if num_class is None:
        raise ValueError("resnet34_dsbn requires either num_class or num_classes.")
    return DSBNResNet(DSBNBasicBlock, [3, 4, 6, 3], num_classes=int(num_class), num_domains=num_domains)


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_dsbn_compatible_state_dict(model: nn.Module, checkpoint) -> dict:
    source_state = _unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    matched = {}
    direct_match_count = 0
    replicated_bn_count = 0
    remapped_downsample_count = 0

    def _candidate_keys_for_dsbn_bn(key: str):
        candidates = []
        if ".bns." in key:
            prefix, rest = key.split(".bns.", 1)
            rest_parts = rest.split(".", 1)
            suffix = rest_parts[1] if len(rest_parts) == 2 else ""
            candidate = prefix if not suffix else f"{prefix}.{suffix}"
            candidates.append(candidate)

            if ".downsample.bn" in prefix:
                ds_prefix = prefix.replace(".downsample.bn", ".downsample.1")
                candidates.append(ds_prefix if not suffix else f"{ds_prefix}.{suffix}")
        return candidates

    def _candidate_keys_for_dsbn_conv(key: str):
        candidates = []
        if ".downsample.conv.weight" in key:
            candidates.append(key.replace(".downsample.conv.weight", ".downsample.0.weight"))
        return candidates

    for key, target_tensor in model_state.items():
        if key in source_state and source_state[key].shape == target_tensor.shape:
            matched[key] = source_state[key]
            direct_match_count += 1
            continue

        found = False
        for candidate in _candidate_keys_for_dsbn_bn(key):
            if candidate in source_state and source_state[candidate].shape == target_tensor.shape:
                matched[key] = source_state[candidate]
                replicated_bn_count += 1
                if ".downsample.bn" in key:
                    remapped_downsample_count += 1
                found = True
                break
        if found:
            continue

        for candidate in _candidate_keys_for_dsbn_conv(key):
            if candidate in source_state and source_state[candidate].shape == target_tensor.shape:
                matched[key] = source_state[candidate]
                direct_match_count += 1
                remapped_downsample_count += 1
                found = True
                break
        if found:
            continue

    model_state.update(matched)
    model.load_state_dict(model_state)
    return {
        "matched_keys": len(matched),
        "total_model_keys": len(model_state),
        "direct_match_keys": direct_match_count,
        "replicated_bn_keys": replicated_bn_count,
        "remapped_downsample_keys": remapped_downsample_count,
        "unmatched_model_keys": len(model_state) - len(matched),
    }
