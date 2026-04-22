NON_ENTITY_LABEL = 'O'
NON_ENTITY_LABEL_ID = 0
PAD_LABEL = 'X'
PAD_LABEL_ID = -100

OBJECT_START_TOKEN = '<O>'
SUBJECT_START_TOKEN = '<E>'
NONE_REL_LABEL = 'None'
NONE_REL_LABEL_ID = 0

PARTIAL_LABEL = 'PARTIAL'
PARTIAL_LABEL_ID = -1

HGMAF_META_FIELDS = (
    'image_id',
    'image_path',
    'aux_data',
    'evidence_text_emotion_text',
    'evidence_text_entity_text',
    'evidence_text_entity_interp_text',
    'evidence_visual_original_path',
    'evidence_visual_generated_path',
    'evidence_visual_entity_path',
)

HGMAF_EVIDENCE_FIELDS = (
    'evidence_text_emotion',
    'evidence_text_entity',
    'evidence_text_entity_interp',
    'evidence_visual_original',
    'evidence_visual_generated',
    'evidence_visual_entity',
)
