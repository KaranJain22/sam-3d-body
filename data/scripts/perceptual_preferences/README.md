# Perceptual Preference Dataset Schema

This folder defines a pairwise human preference format for supervising per-vertex perceptual weights `omega(v)`.

Each row in a JSONL file represents one pairwise comparison between two candidate meshes for the same image/person.

## JSONL fields

- `sample_id` (`str`): unique preference sample id.
- `image_id` (`str`): source frame id.
- `person_id` (`str|int`): person track/id in the frame.
- `candidate_a` (`object`): metadata/features for mesh A.
- `candidate_b` (`object`): metadata/features for mesh B.
- `preferred` (`"a"|"b"`): human preference label.
- `confidence` (`float`, optional): annotation confidence in `[0, 1]`.

`candidate_a` and `candidate_b` should each contain:

- `mesh_id` (`str`)
- `semantic_part_ids` (`int[V]`)
- `curvature_scale_features` (`float[V][2]`)
- `visibility` (`float[V]`)
- `screen_size` (`float[V]`)
- `vertex_error` (`float[V]`, optional): candidate-specific residual/error used for ranking targets.

## Ranking target

Given per-vertex predicted weights `omega(v)` and candidate errors `e_a(v), e_b(v)`, compute aggregate perceptual score:

`S_x = -sum_v omega(v) * e_x(v)`

Pairwise loss uses Bradley-Terry logistic objective with `preferred` as supervision.
