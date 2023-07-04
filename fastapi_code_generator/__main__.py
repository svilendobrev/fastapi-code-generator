import re
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, List, Optional
from argparse import ArgumentParser

from datamodel_code_generator import LiteralType, PythonVersion, chdir
from datamodel_code_generator.format import CodeFormatter
from datamodel_code_generator.imports import Import, Imports
from datamodel_code_generator.reference import Reference
from datamodel_code_generator.types import DataType
from jinja2 import Environment, FileSystemLoader

from fastapi_code_generator.parser import OpenAPIParser
from fastapi_code_generator.visitor import Visitor


TITLE_PATTERN = re.compile(r'(?<!^)(?<![A-Z ])(?=[A-Z])| ')
BUILTIN_MODULAR_TEMPLATE_DIR = Path(__file__).parent / "modular_template"
BUILTIN_TEMPLATE_DIR = Path(__file__).parent / "template"
BUILTIN_VISITOR_DIR = Path(__file__).parent / "visitors"
MODEL_PATH: Path = Path("models.py")


def dynamic_load_module(module_path: Path) -> Any:
    module_name = module_path.stem
    spec = spec_from_file_location(module_name, str(module_path))
    if spec:
        module = module_from_spec(spec)
        if spec.loader:
            spec.loader.exec_module(module)
            return module
    raise Exception(f"{module_name} can not be loaded")


def main(
#    input_file: typer.FileText = typer.Option(..., "--input", "-i"),
#    output_dir: Path = typer.Option(..., "--output", "-o"),
#    model_file: str = typer.Option(None, "--model-file", "-m"),
#    template_dir: Optional[Path] = typer.Option(None, "--template-dir", "-t"),
#    enum_field_as_literal: Optional[LiteralType] = typer.Option( None, "--enum-field-as-literal"),
#    generate_routers: bool = typer.Option(False, "--generate-routers", "-r"),
#    specify_tags: Optional[str] = typer.Option(None, "--specify-tags"),
#    custom_visitors: Optional[List[Path]] = typer.Option( None, "--custom-visitor", "-c"),
#    disable_timestamp: bool = typer.Option(False, "--disable-timestamp"),
) -> None:
    #from datamodel_code_generator.__main__ import arg_parser as dcg_ap
    #dcg_ap.add_help=False
    #TODO.. group/separate dcg_ap.argz, remove irrelevant like --input etc
    ap = ArgumentParser() # parents= [ dcg_ap ], conflict_handler= 'resolve')

    def argany( name, *short, **ka):
        return ap.add_argument( dest=name, *(list(short)+['--'+name.replace('_','-')] ), **ka)
    argtext = argany
    def argpath( name, *short, **ka):
        return argany( name, type=Path, *short,**ka)
    def argbool( name, *short, **ka):
        return argany( name, action='store_true', *short,**ka)
    argpath( 'input_file', '-i', required=True )    #Path
    argpath( 'output_dir', '-o', required=True )    #Path
    argtext( 'model_file', '-m', default= MODEL_PATH )
    argpath( 'template_dir', '-t')   #Optional[Path]
    argtext( 'enum_field_as_literal', choices = list( v.value for v in LiteralType )) # Optional[LiteralType]
    argbool( 'generate_routers', '-r')
    argtext( 'specify_tags')
    argpath( 'custom_visitor', '-c', action='append' )
    argbool( 'disable_timestamp')
    argz = ap.parse_args()

    return generate_code(
        input_name              = argz.input_file,
        input_text              = open( argz.input_file ).read(),
        output_dir              = argz.output_dir,
        template_dir            = argz.template_dir,
        model_path              = argz.model_file and Path( argz.model_file ).with_suffix('.py'),
        custom_visitors         = argz.custom_visitor or (),
        disable_timestamp       = argz.disable_timestamp,
        generate_routers        = argz.generate_routers,
        specify_tags            = argz.specify_tags,
        enum_field_as_literal   = argz.enum_field_as_literal and LiteralType( argz.enum_field_as_literal ),
    )


def _get_most_of_reference(data_type: DataType) -> Optional[Reference]:
    if data_type.reference:
        return data_type.reference
    for data_type in data_type.data_types:
        reference = _get_most_of_reference(data_type)
        if reference:
            return reference
    return None


def generate_code(
    input_name: str,
    input_text: str,
    output_dir: Path,
    template_dir: Optional[Path],
    model_path: Optional[Path] = None,
    enum_field_as_literal: Optional[str] = None,
    custom_visitors: Optional[List[Path]] = [],
    disable_timestamp: bool = False,
    generate_routers: Optional[bool] = None,
    specify_tags: Optional[str] = None,
) -> None:
    if not model_path:
        model_path = MODEL_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    if generate_routers:
        template_dir = BUILTIN_MODULAR_TEMPLATE_DIR
        Path(output_dir / "routers").mkdir(parents=True, exist_ok=True)
    if not template_dir:
        template_dir = BUILTIN_TEMPLATE_DIR
    parser = OpenAPIParser(input_text, enum_field_as_literal=enum_field_as_literal)
    with chdir(output_dir):
        models = parser.parse()
    if not models:
        return
    elif isinstance(models, str):
        output = output_dir / model_path
        modules = {output: (models, input_name)}
    else:
        raise Exception('Modular references are not supported in this version')

    environment = Environment(
        loader=FileSystemLoader(
            template_dir if template_dir else f"{Path(__file__).parent}/template",
            encoding="utf8",
        ),
    )

    results: Dict[Path, str] = {}
    code_formatter = CodeFormatter(PythonVersion.PY_38, Path().resolve())

    template_vars = {"info": parser.parse_info()}
    visitors = []

    # Load visitors
    builtin_visitors = BUILTIN_VISITOR_DIR.rglob("*.py")
    visitors_path = [*builtin_visitors, *custom_visitors]
    for visitor_path in visitors_path:
        module = dynamic_load_module(visitor_path)
        try:
            visitors.append(module.visit)
        except AttributeError:
            raise RuntimeError(f"{visitor_path.stem} does not have any visit function")

    # Call visitors to build template_vars
    for visitor in visitors:
        template_vars.update( visitor(parser, model_path))

    all_tags = []
    if generate_routers:
        for operation in template_vars.get("operations", ()):
            all_tags += getattr( operation, 'tags', ())

    # Convert from Tag Names to router_names
    sorted_tags = sorted(set(all_tags))
    routers = sorted(
        re.sub(TITLE_PATTERN, '_', tag.strip()).lower() for tag in sorted_tags
    )
    template_vars.update( routers=routers, tags=sorted_tags)

    for target in template_dir.rglob("*"):
        relative_path = target.relative_to(template_dir)
        template = environment.get_template(str(relative_path))
        result = template.render(template_vars)
        results[relative_path] = code_formatter.format_code(result)

    if generate_routers:
        tags = sorted_tags
        results.pop(Path("routers.jinja2"))
        if specify_tags:
            if Path(output_dir.joinpath("main.py")).exists():
                with open(Path(output_dir.joinpath("main.py")), 'r') as file:
                    content = file.read()
                    if "app.include_router" in content:
                        tags = sorted(
                            set(tag.strip() for tag in str(specify_tags).split(","))
                        )

        for target in BUILTIN_MODULAR_TEMPLATE_DIR.rglob("routers.*"):
            relative_path = target.relative_to(template_dir)
            for router, tag in zip(routers, sorted_tags):
                if (
                    not Path(output_dir.joinpath("routers", router))
                    .with_suffix(".py")
                    .exists()
                    or tag in tags
                ):
                    template_vars["tag"] = tag.strip()
                    template = environment.get_template(str(relative_path))
                    result = template.render(template_vars)
                    router_path = Path("routers", router).with_suffix(".jinja2")
                    results[router_path] = code_formatter.format_code(result)

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    header = f"""\
# generated by fastapi-codegen:
#   filename:  {Path(input_name).name}"""
    if not disable_timestamp:
        header += f"\n#   timestamp: {timestamp}"

    for path, code in results.items():
        with output_dir.joinpath(path.with_suffix(".py")).open("wt") as file:
            print(header, file=file)
            print("", file=file)
            print(code.rstrip(), file=file)

    header = f'''\
# generated by fastapi-codegen:
#   filename:  {{filename}}'''
    if not disable_timestamp:
        header += f'\n#   timestamp: {timestamp}'

    class wither:
        'empty context for with operator - i.e. with (somefile or wither): ...'
        def __enter__(*a,**k): pass
        def __exit__(*a,**k): pass
    wither=wither()

    for path, body_and_filename in modules.items():
        body, filename = body_and_filename
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        with ( path.open('wt', encoding='utf8') if path else wither) as file:
            print(header.format(filename=filename), file=file)
            if body:
                print('', file=file)
                print(body.rstrip(), file=file)


if __name__ == "__main__":
    main()

# vim:ts=4:sw=4:expandtab
