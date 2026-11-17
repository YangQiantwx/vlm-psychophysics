"""The archived native-final parser shared by Qwen and Nemotron."""

import re


def parse_generated_response(response, tokenizer, generation, allowed, *, legacy_parser_version):
    """Score only a complete final channel; never extract labels from reasoning."""
    token_ids=list(response.token_ids)
    raw=response.text
    exact=tokenizer.decode(token_ids,skip_special_tokens=False)
    final=raw; reasoning=None; complete=None; split_status='thinking_disabled'
    reasoning_span=final_span=boundary_tokens=None
    if generation['enable_thinking']:
        closing=tokenizer.encode('</think>',add_special_tokens=False)
        assert closing, 'Native reasoning delimiter must tokenize'
        positions=[i for i in range(len(token_ids)-len(closing)+1)
                   if token_ids[i:i+len(closing)]==closing]
        final=None; complete=False
        if len(positions)==1:
            boundary=positions[0]
            reasoning_span=boundary
            boundary_tokens=len(closing)
            final_span=len(token_ids)-boundary-boundary_tokens
            reasoning=tokenizer.decode(token_ids[:boundary],skip_special_tokens=False)
            if reasoning.startswith('<think>'): reasoning=reasoning[len('<think>'):]
            # EOS removal is tokenizer-defined. No text label is searched for.
            final_ids=token_ids[boundary+len(closing):]
            final=tokenizer.decode(final_ids,skip_special_tokens=True)
            complete=True; split_status='native_think_end_token'
            opening=tokenizer.encode('<think>',add_special_tokens=False)
            if opening and any(final_ids[i:i+len(opening)]==opening
                               for i in range(len(final_ids)-len(opening)+1)):
                final=None; complete=False; split_status='reopened_thinking_in_final'
        else:
            reasoning=exact
            split_status='missing_think_end' if not positions else 'multiple_think_end'
    final_channel_text=final
    if generation['enable_thinking'] and response.finish_reason!='stop':
        final=None
    normalized=(re.sub(r'[.!。！]$','',final.strip()).strip().upper()
                if final is not None else None)
    semantic=normalized if normalized in allowed else None
    strict=final is not None and final.strip() in allowed
    # A non-stop reasoning run has not completed its answer, even if its last
    # generated token forms a label. The raw and partial final channel remain.
    truncated=response.finish_reason=='length'
    return {'raw_answer':raw,'raw_output_with_special_tokens':exact,
        'reasoning_text':reasoning,'final_answer':final,'final_channel_text':final_channel_text,
        'reasoning_complete':complete,'generation_complete':response.finish_reason=='stop',
        'reasoning_parse_status':split_status,'raw_format_valid':raw.strip() in allowed,
        'semantic_parsed_answer':semantic,'format_valid':bool(strict),
        'parser_version':('native_think_boundary_then_single_label_v2' if generation['enable_thinking']
                          else legacy_parser_version),
        'finish_reason':response.finish_reason,'stop_reason':response.stop_reason,
        'output_truncated':truncated,'output_token_ids':token_ids,'output_tokens':len(token_ids),
        'reasoning_tokens':None,
        'native_reasoning_span_tokens':reasoning_span,'native_final_span_tokens':final_span,
        'native_reasoning_boundary_tokens':boundary_tokens,
        'native_span_token_count_method':'Derived from raw generated token-ID positions around the unique native </think> delimiter; spans exclude that delimiter but include any emitted opening/EOS tokens. Not API reasoning usage.',
        'reasoning_tokens_note':'Offline vLLM CompletionOutput does not expose reasoning-token usage; no estimate substituted.'}
