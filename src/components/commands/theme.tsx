import type { CommandComponentProps, ComponentCommand } from '@commands';
import Usage from '@components/terminal-usage';
import checkThemeSwitch from '@fn/check-theme-switch';
import isArgumentInvalid from '@fn/is-argument-invalid';
import useHistoryState from '@history';
import useRerenderState from '@rerender';
import useThemeState from '@theme';
import split from 'lodash/split';
import type { FunctionalComponent } from 'preact';

const Theme: FunctionalComponent<CommandComponentProps> = ({ args: commandArguments = [] }) => {
	const { rerender } = useRerenderState();
	const { history } = useHistoryState();
	const { setTheme, themes } = useThemeState();

	/* ===== get current command ===== */
	const currentCommand = split(history[0], ' ');

	if (checkThemeSwitch(rerender, currentCommand, themes)) {
		const current = currentCommand[2];
		if (current !== undefined) {
			setTheme(current);
		}
	}

	/* ===== check arg is valid ===== */
	const checkArgument = () => (isArgumentInvalid(commandArguments, 'set', themes) ? <Usage cmd='themes' /> : <></>);

	if (commandArguments.length > 0 || commandArguments.length > 2) {
		return checkArgument();
	}

	return (
		<div
			className='terminal-line-history'
			data-testid='themes'>
			<h3>Available Themes</h3>
			<div
				style={{
					display: 'grid',
					gap: '1rem',
					gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
					marginBottom: '1rem',
				}}>
				{themes.map(theme => (
					<div
						key={theme}
						onClick={() => setTheme(theme)}
						onMouseEnter={e => {
							e.currentTarget.style.transform = 'translateY(-2px)';
							e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
						}}
						onMouseLeave={e => {
							e.currentTarget.style.transform = 'translateY(0)';
							e.currentTarget.style.boxShadow = 'none';
						}}
						style={{
							background: 'var(--color-bg-100)',
							border: '1px solid var(--color-border)',
							borderRadius: '8px',
							cursor: 'pointer',
							padding: '1rem',
							transition: 'all 0.3s ease',
						}}>
						<div style={{ fontWeight: 'bold', marginBottom: '0.5rem', textTransform: 'capitalize' }}>
							{theme}
						</div>
						<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>
							{theme === 'dark' && 'Classic dark theme with high contrast'}
							{theme === 'light' && 'Clean light theme for daytime use'}
							{theme === 'terminal' && 'Retro terminal green on black'}
							{theme === 'neon' && 'Vibrant neon colors for night mode'}
							{theme === 'minimal' && 'Minimalist design with subtle colors'}
						</div>
					</div>
				))}
			</div>
			<Usage
				cmd='themes'
				marginY
			/>
		</div>
	);
};

const ThemeCommand: ComponentCommand = {
	command: 'themes',
	component: Theme,
};

export default ThemeCommand;
