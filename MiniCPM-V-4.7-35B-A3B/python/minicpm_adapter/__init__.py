def register():
    from vllm import ModelRegistry
    from vllm.reasoning import ReasoningParserManager
    from vllm.parser.parser_manager import ParserManager
    ModelRegistry.register_model("MiniCPMV4_7ForConditionalGeneration", "minicpm_adapter.model:MiniCPMV4_7ForConditionalGeneration")
    ReasoningParserManager.register_lazy_module("minicpmv", "minicpm_adapter.reasoning", "MiniCPMVParserReasoningAdapter")
    if getattr(ParserManager, "_hks_minicpm47", False):
        return
    original = ParserManager.get_parser.__func__
    @classmethod
    def get_parser(cls, *args, **kwargs):
        import inspect
        bound=inspect.signature(original).bind(cls,*args,**kwargs)
        bound.apply_defaults()
        base=original(cls,*args,**kwargs)
        if bound.arguments.get("reasoning_parser_name") != "minicpmv" or base is None:
            return base
        from .parser import MiniCPMVUnifiedParser
        class MiniCPMLocalParser(MiniCPMVUnifiedParser):
            reasoning_parser_cls=base.reasoning_parser_cls
            tool_parser_cls=base.tool_parser_cls
        return MiniCPMLocalParser
    ParserManager.get_parser=get_parser
    ParserManager._hks_minicpm47=True
