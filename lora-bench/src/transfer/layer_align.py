"""Layer alignment teacher → student via linear CKA.

For each student layer j, pick teacher layer i = argmax CKA(h_s^j, h_t^i)
over the same calibration tokens. Falls back to linear stride mapping
if max CKA below threshold (configurable).
"""
from __future__ import annotations

import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between activations X (N,d_x) and Y (N,d_y). Both centered."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    num = (Y.T @ X).pow(2).sum()
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    if den <= 0:
        return 0.0
    return float(num / den)


def cka_layer_map(
    teacher_hidden: list[torch.Tensor],
    student_hidden: list[torch.Tensor],
    min_cka: float = 0.05,
) -> list[int]:
    """Return list of length L_s giving teacher layer index for each student layer.

    teacher_hidden[i]: (N, d_t) flattened token activations at teacher layer i.
    student_hidden[j]: (N, d_s) idem student.
    Falls back to linear stride for layers where best CKA < min_cka.
    """
    L_t = len(teacher_hidden)
    L_s = len(student_hidden)
    mapping = []
    for j in range(L_s):
        best_i, best_cka = -1, -1.0
        for i in range(L_t):
            c = linear_cka(student_hidden[j], teacher_hidden[i])
            if c > best_cka:
                best_cka, best_i = c, i
        if best_cka < min_cka:
            best_i = round(j * (L_t - 1) / max(L_s - 1, 1))
        mapping.append(best_i)
    return mapping


def linear_layer_map(L_s: int, L_t: int) -> list[int]:
    """Fallback: stride mapping student layer j → teacher round(j * L_t / L_s)."""
    return [round(j * (L_t - 1) / max(L_s - 1, 1)) for j in range(L_s)]
