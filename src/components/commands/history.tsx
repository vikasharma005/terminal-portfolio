import type { ComponentCommand } from '@commands';
import useHistoryState from '@history';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('history', {
	clear: 'Clear History',
	commands: 'Commands executed:',
	noHistory: 'No commands in history yet.',
	title: 'Command History',
});

const History: FunctionalComponent = () => {
	const t = useStore(messages);
	const { clearHistory, history } = useHistoryState();

	const handleClearHistory = () => {
		if (confirm('Are you sure you want to clear the command history?')) {
			clearHistory();
		}
	};

	return (
		<div className='terminal-line-history'>
			<div
				style={{
					alignItems: 'center',
					display: 'flex',
					justifyContent: 'space-between',
					marginBottom: '1rem',
				}}>
				<h3>{t.title}</h3>
				<button
					onClick={handleClearHistory}
					onMouseEnter={e => {
						e.currentTarget.style.background = 'var(--color-primary-hover)';
					}}
					onMouseLeave={e => {
						e.currentTarget.style.background = 'var(--color-primary)';
					}}
					style={{
						background: 'var(--color-primary)',
						border: 'none',
						borderRadius: '4px',
						color: 'white',
						cursor: 'pointer',
						fontSize: '0.9rem',
						padding: '0.5rem 1rem',
						transition: 'background 0.3s ease',
					}}>
					{t.clear}
				</button>
			</div>

			{history.length === 0 ? (
				<p style={{ color: 'var(--color-text-200)', fontStyle: 'italic' }}>{t.noHistory}</p>
			) : (
				<div>
					<p style={{ color: 'var(--color-text-200)', marginBottom: '1rem' }}>
						{t.commands} <strong>{history.length}</strong>
					</p>
					<div
						style={{
							background: 'var(--color-bg-100)',
							border: '1px solid var(--color-border)',
							borderRadius: '4px',
							maxHeight: '300px',
							overflowY: 'auto',
							padding: '1rem',
						}}>
						{history.map((command, index) => (
							<div
								key={index}
								style={{
									borderBottom: index < history.length - 1 ? '1px solid var(--color-border)' : 'none',
									fontFamily: 'monospace',
									fontSize: '0.9rem',
									padding: '0.5rem 0',
								}}>
								<span style={{ color: 'var(--color-text-200)', marginRight: '1rem' }}>
									{String(index + 1).padStart(3, '0')}
								</span>
								<span style={{ color: 'var(--color-primary)' }}>$</span>
								<span style={{ marginLeft: '0.5rem' }}>{command}</span>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
};

const HistoryCommand: ComponentCommand = {
	command: 'history',
	component: History,
};

export default HistoryCommand;
