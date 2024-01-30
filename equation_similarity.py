from bs4 import BeautifulSoup
import lxml.etree as etree

def extract_variables_from_mathml(html_equation):
    soup = BeautifulSoup(html_equation, 'html.parser')
    mathml = soup.find('math')

    if mathml:
        xml_content = str(mathml)
        root = etree.fromstring(xml_content)

        variables = set()

        # XPath expression to find all identifiers in MathML
        identifier_elements = root.xpath('//m:mi', namespaces={'m': 'http://www.w3.org/1998/Math/MathML'})

        for identifier in identifier_elements:
            variables.add(identifier.text)

        return list(variables)

    return []

# Example usage:
html_equation = "<math><mi>x</mi><mo>+</mo><mi>y</mi></math>"
variables = extract_variables_from_mathml(html_equation)
print(variables)
