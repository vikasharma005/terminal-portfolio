import type { ComponentCommand } from '@commands';
import type { FunctionalComponent } from 'preact';
import { useEffect, useState } from 'preact/hooks';

const Exit: FunctionalComponent = () => {
	const [countdown, setCountdown] = useState(5);
	const [isClosing, setIsClosing] = useState(false);

	// Disable terminal input during countdown
	useEffect(() => {
		const terminalInput = document.querySelector('input[type="text"]')!;
		if (terminalInput) {
			terminalInput.disabled = true;
			terminalInput.placeholder = 'Terminal is closing...';
		}

		// Re-enable input if component unmounts before countdown ends
		return () => {
			if (terminalInput) {
				terminalInput.disabled = false;
				terminalInput.placeholder = '';
			}
		};
	}, []);

	useEffect(() => {
		if (countdown > 0) {
			const timer = setTimeout(() => {
				setCountdown(countdown - 1);
			}, 1000);

			return () => clearTimeout(timer);
		}
		setIsClosing(true);
		// Try to close the window/tab
		setTimeout(() => {
			// Try multiple methods to close the tab
			if (window.opener) {
				window.close();
			} else {
				// For modern browsers, try to close the tab
				window.close();
				// If that doesn't work, try to navigate back and close
				if (globalThis.history.length > 1) {
					globalThis.history.back();
					setTimeout(() => window.close(), 100);
				}
			}
		}, 500);
	}, [countdown]);

	return (
		<div className='terminal-line-history'>
			<div
				style={{
					background: 'var(--color-bg-100)',
					border: '2px solid var(--color-primary)',
					borderRadius: '8px',
					margin: '1rem 0',
					padding: '2rem',
					textAlign: 'center',
				}}>
				<h3 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>
					Goodbye! Thanks for visiting my portfolio.
				</h3>
				<div
					style={{
						color: countdown <= 3 ? '#F44336' : 'var(--color-text)',
						fontFamily: 'monospace',
						fontSize: '2rem',
						fontWeight: 'bold',
						marginBottom: '1rem',
					}}>
					{isClosing ? 'Closing...' : `Closing in ${countdown} seconds`}
				</div>
				<div
					style={{
						background: 'var(--color-bg-200)',
						borderRadius: '2px',
						height: '4px',
						overflow: 'hidden',
						width: '100%',
					}}>
					<div
						style={{
							background: 'var(--color-primary)',
							height: '100%',
							transition: 'width 1s linear',
							width: `${((5 - countdown) / 5) * 100}%`,
						}}
					/>
				</div>
				<p style={{ color: 'var(--color-text-200)', fontSize: '0.9rem', marginTop: '1rem' }}>
					{isClosing
						? 'Thank you for exploring my terminal portfolio!'
						: 'The terminal will close automatically...'}
				</p>
			</div>
		</div>
	);
};

const ExitCommand: ComponentCommand = {
	command: 'exit',
	component: Exit,
};

export default ExitCommand;
